import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vectorizer = None
mission_encoder = None

def rollout_one_task(args):
    (make_env_fn, mission, policy_cls, policy_kwargs, 
     policy_state_dict,adapted_params_cpu, batch_size, gamma) = args

    env = make_env_fn()
    env.reset_task(mission)

    policy = policy_cls(**policy_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    obs_list, act_list, rew_list, ep_id_list = [], [], [], []
    total_steps = 0

    for ep in range(batch_size):
        obs, info = env.reset()
        done = False; steps = 0
        while not done:
            obs_vec = preprocess_obs(obs)
            with torch.no_grad():
                pi = policy(torch.from_numpy(obs_vec[None, :]).float(), 
                            params=adapted_params_cpu)
                action = pi.sample().item()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            obs_list.append(obs_vec)
            act_list.append(action)
            rew_list.append(r)
            ep_id_list.append(ep)
        total_steps += steps

    return (mission, total_steps, obs_list, act_list, rew_list, ep_id_list)




# Mission Wrapper
class BabyAIMissionTaskWrapper(gym.Wrapper):
    def __init__(self, env, missions=None):
        assert missions is not None, "tasks not there"
        super().__init__(env)
        self.missions = missions
        self.current_mission = None

    def sample_tasks(self, n_tasks):
        return list(np.random.choice(self.missions, n_tasks, replace=False))

    def reset_task(self, mission):
        self.current_mission = mission
        if hasattr(self.env, 'set_forced_mission'):
            self.env.set_forced_mission(mission)

    def reset(self, **kwargs):        
        result = super().reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        if self.current_mission is not None:
            obs['mission'] = self.current_mission
        if isinstance(result, tuple):
            return obs, info
        else:
            return obs
        

# Mission Encoder
class MissionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# MissionParamAdapter 
class MissionParamAdapter(nn.Module):
    def __init__(self, mission_adapter_input_dim, policy_param_shapes):
        super().__init__()
        self.policy_param_shapes = policy_param_shapes
        total_params = sum([torch.Size(shape).numel() for shape in policy_param_shapes])
        self.net = nn.Sequential(
            nn.Linear(mission_adapter_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params),
            nn.Tanh()  
        )
    def forward(self, mission_emb):

        out = self.net(mission_emb)  
        split_points = []
        total = 0
        for shape in self.policy_param_shapes:
            num = torch.Size(shape).numel()
            split_points.append(total + num)
            total += num
        chunks = torch.split(out, [torch.Size(shape).numel() for shape in self.policy_param_shapes], dim=1)
        reshaped = [chunk.view(-1, *shape) for chunk, shape in zip(chunks, self.policy_param_shapes)]
        return reshaped 
     

def preprocess_obs(obs):

    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    
    return np.concatenate([image, direction])
    

# Sampler
class MultiTaskSampler(object):
    def __init__(self,
                 env=None,   
                 env_fn=None,      
                 batch_size=None,        
                 policy=None,
                 baseline=None,
                 seed=None,
                 num_workers=0):   
        assert env is not None, "Must pass BabyAI env"
        self.env = env
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers

    def sample_tasks(self, num_tasks):
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, meta_learner, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):

        tasks = self.sample_tasks(meta_batch_size)  
        all_step_counts = []
        valid_episodes_all = []
        if (self.num_workers or 0) > 0:
            assert self.env_fn is not None, "env_fn no there "
 
            policy_state_dict_cpu = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            policy_cls = self.policy.__class__
            policy_kwargs = dict(
                input_size=self.policy.input_size,
                output_size=self.policy.output_size,
                hidden_sizes=self.policy.hidden_sizes,
                nonlinearity=self.policy.nonlinearity
            )

            tasks = self.sample_tasks(meta_batch_size)

            # compute theta' per task
            adapted_params_cpu = []
            for t in tasks:
                theta_prime = meta_learner.adapt_one(t)
                adapted_params_cpu.append({k: v.detach().cpu() for k, v in theta_prime.items()})

            worker_args = []
            for t, p in zip(tasks, adapted_params_cpu):
                worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                    policy_state_dict_cpu, p, self.batch_size, gamma))

            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(ex.map(rollout_one_task, worker_args))

            valid_episodes_all, all_step_counts = [], []
            for (mission, step_count, obs_list, act_list, rew_list, ep_list) in results:
                be = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                be.mission = mission
                for o, a, r, e in zip(obs_list, act_list, rew_list, ep_list):
                    be.append([o], [np.array(a)], [np.array(r)], [np.array(e)])
                self.baseline.fit(be)
                be.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                valid_episodes_all.append(be)
                all_step_counts.append(step_count)

            return (valid_episodes_all, all_step_counts)