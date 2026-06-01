
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym 
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss
import random
import builtins, io
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def silence_sampling_rejected():
    real_print = builtins.print
    buf = io.StringIO()
    def filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("Sampling rejected: unreachable object"):
            return
        return real_print(*args, **kwargs)
    builtins.print = filtered_print
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            yield
    finally:
        builtins.print = real_print


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs):
    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]
    return np.concatenate([image, direction])  


def update_head_only(policy, loss, params=None, step_size=0.5, first_order=True, head_layer_prefix='layer3', debug=False):

    if params is None:
        params = OrderedDict(policy.named_parameters())
    
    # Store original values for verification (only if debug)
    if debug:
        original_params = {k: v.clone().detach() for k, v in params.items()}
    
    # Identify head parameters
    head_params = OrderedDict(
        (name, param) for name, param in params.items() 
        if head_layer_prefix in name
    )
    
    # Compute gradients only for head parameters
    grads = torch.autograd.grad(loss, head_params.values(),
                               create_graph=not first_order)
    
    # Build updated params: update head, keep body frozen
    updated_params = OrderedDict()
    grad_iter = iter(grads)
    
    for name, param in params.items():
        if head_layer_prefix in name:
            # Update head parameters
            grad = next(grad_iter)
            updated_params[name] = param - step_size * grad
        else:
            # Keep body parameters frozen (just copy reference)
            updated_params[name] = param
    
    # Verification logging (only if debug)
    if debug:
        print("\n  [ANIL Inner Loop Verification]")
        for name in params.keys():
            orig = original_params[name]
            updated = updated_params[name].detach()
            diff = (updated - orig).abs().max().item()
            if head_layer_prefix in name:
                print(f"    {name}: diff={diff:.6f} -> GRADIENT APPLIED")
            else:
                print(f"    {name}: diff={diff:.6f} -> FROZEN (no grad)")
    
    return updated_params


def rollout_one_task_anil(args):
    """Worker function for parallel ANIL rollouts."""
    (make_env_fn, task, policy_cls, policy_kwargs,
     policy_state_dict, adapted_params_cpu, batch_size, gamma) = args
    
    # Suppress warnings in worker processes
    import warnings
    import logging
    import os
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    logging.getLogger("gym").setLevel(logging.ERROR)
    logging.getLogger("gymnasium").setLevel(logging.ERROR)

    env = make_env_fn()
    env.reset_task(task)

    policy = policy_cls(**policy_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    obs_list, action_list, reward_list, episode_id_list = [], [], [], []
    total_steps = 0

    for episode in range(batch_size):
        with silence_sampling_rejected():
            obs, info = env.reset()
        done, steps = False, 0
        while not done:
            obs_vec = preprocess_obs(obs)
            with torch.no_grad():
                pi = policy(torch.from_numpy(obs_vec[None, :]).float(),
                            params=adapted_params_cpu)
                action = pi.sample().item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            obs_list.append(obs_vec)
            action_list.append(action)
            reward_list.append(reward)
            episode_id_list.append(episode)
        total_steps += steps

    return (task, total_steps, obs_list, action_list, reward_list, episode_id_list)


# Mission Wrapper (same as MAML)
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


class ANILMultiTaskSampler(object):
    """
    ANIL-style Multi-Task Sampler
    
    Key difference from MAML's MultiTaskSampler:
    - Inner loop only updates HEAD parameters (last layer)
    - BODY parameters (feature extractor) remain frozen during adaptation
    """
    
    def __init__(self,    
                 env=None, 
                 env_fn=None,               
                 batch_size=None,       
                 policy=None,
                 baseline=None,     
                 seed=None,
                 num_workers=0,
                 head_layer_prefix='layer3'):  # ANIL-specific: which layer is the head
                 
        assert env is not None, "Must pass BabyAI env"
        self.env = env
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers
        self.head_layer_prefix = head_layer_prefix
        
        # Log ANIL configuration
        self._log_anil_config()
    
    def _log_anil_config(self):
        """Log which parameters will be updated in inner loop."""
        head_params = []
        body_params = []
        for name, _ in self.policy.named_parameters():
            if self.head_layer_prefix in name:
                head_params.append(name)
            else:
                body_params.append(name)
        print(f"[ANIL Sampler] Head (updated in inner loop): {head_params}")
        print(f"[ANIL Sampler] Body (frozen in inner loop): {body_params}")

    def sample_tasks(self, num_tasks):
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu', debug=False):
        """
        Sample trajectories for meta-learning with ANIL-style inner loop.
        
        Key difference from MAML:
        - Inner loop only updates head parameters
        - Body stays frozen during task adaptation
        
        Args:
            debug: If True, print verification that body stays frozen
        """
        tasks = self.sample_tasks(meta_batch_size)  

        train_episodes_all = []
        valid_episodes_all = []
        all_step_counts = []  

        if (self.num_workers or 0) == 0:
            # Single-process execution
            for task_index, task in enumerate(tasks):
                self.env.reset_task(task)
                train_batches = []
                params = None
                
                for step in range(num_steps):
                    batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                    
                    for episode in range(self.batch_size):
                        with silence_sampling_rejected():
                            obs, info = self.env.reset()
                        done = False

                        episode_obs = []
                        episode_actions = []
                        episode_rewards = []
                        step_count = 0

                        while not done:
                            obs_vec = preprocess_obs(obs)
                            if np.isnan(obs_vec).any():
                                print("NaN in obs_vec, skipping episode")
                                break
                            obs_tensor = np.expand_dims(obs_vec, axis=0)
                            obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                            
                            with torch.no_grad():
                                pi = self.policy(obs_tensor, params=params)
                                action = pi.sample().item()

                            if np.isnan(action):
                                print("NaN in action, skipping episode")
                                break
                            
                            obs, reward, terminated, truncated, info = self.env.step(action)
                            step_count += 1

                            if np.isnan(reward):
                                print("NaN in reward, skipping episode")
                                break

                            done = terminated or truncated
                            episode_obs.append(obs_vec)
                            episode_actions.append(action)
                            episode_rewards.append(reward)

                        all_step_counts.append(step_count)  
                        
                        if len(episode_obs) > 0 and not np.isnan(episode_obs).any():
                            batch.append(
                                episode_obs,
                                [np.array(a) for a in episode_actions],
                                [np.array(r) for r in episode_rewards],
                                [episode]*len(episode_obs)
                            )
                                                
                    self.baseline.fit(batch)
                    batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                    
                    if torch.isnan(batch.advantages).any():
                        print("NaN in batch advantages!")
                    if torch.isnan(batch.observations).any():
                        print("NaN in batch observations!")

                    # ANIL: Compute loss and update ONLY HEAD parameters
                    loss = reinforce_loss(self.policy, batch, params=params)
                    params = update_head_only(
                        self.policy, 
                        loss, 
                        params=params, 
                        step_size=fast_lr, 
                        first_order=True,
                        head_layer_prefix=self.head_layer_prefix,
                        debug=debug  # Pass debug flag for verification
                    )
                    train_batches.append(batch)
                    
                train_episodes_all.append(train_batches)

                # Validation rollout (using adapted params - with only head updated)
                valid_batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                for episodes in range(self.batch_size):
                    obs, info = self.env.reset()
                    done = False
                    while not done:
                        obs_vec = preprocess_obs(obs)
                        obs_tensor = np.expand_dims(obs_vec, axis=0)
                        obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                        with torch.no_grad():
                            pi = self.policy(obs_tensor, params=params)
                            action = pi.sample().item()
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        valid_batch.append([obs_vec], [np.array(action)], [np.array(reward)], [episodes])
                
                self.baseline.fit(valid_batch)
                valid_batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                valid_episodes_all.append(valid_batch)

        else:
            # Multi-worker execution
            assert self.env_fn is not None, "env_fn not there"

            policy_state_dict_cpu = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            policy_cls = self.policy.__class__
            policy_kwargs = dict(
                input_size=self.policy.input_size,
                output_size=self.policy.output_size,
                hidden_sizes=self.policy.hidden_sizes,
                nonlinearity=self.policy.nonlinearity
            )

            task_params = [None for _ in tasks]
            per_task_train_batches = [[] for _ in tasks]

            for _ in range(num_steps):
                worker_args = []
                for t, p in zip(tasks, task_params):
                    p_cpu = None if p is None else {k: v.detach().cpu() for k, v in p.items()}
                    worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                        policy_state_dict_cpu, p_cpu, self.batch_size, gamma))
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                        results = list(ex.map(rollout_one_task_anil, worker_args))

                # Build batches and update only head
                new_task_params = []
                for task_idx, (mission, step_count, obs_list, act_list, rew_list, ep_list) in enumerate(results):
                    batch_episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                    for obs, action, reward, episode in zip(obs_list, act_list, rew_list, ep_list):
                        batch_episodes.append([obs], [np.array(action)], [np.array(reward)], [np.array(episode)])
                    self.baseline.fit(batch_episodes)
                    batch_episodes.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                    per_task_train_batches[task_idx].append(batch_episodes)
                    all_step_counts.append(step_count)

                    # ANIL: Compute inner loss & update ONLY HEAD (θ_head → θ'_head)
                    loss = reinforce_loss(self.policy, batch_episodes, params=task_params[task_idx])
                    theta_prime = update_head_only(
                        self.policy,
                        loss,
                        params=task_params[task_idx],
                        step_size=fast_lr,
                        first_order=True,
                        head_layer_prefix=self.head_layer_prefix,
                        debug=debug  # Pass debug flag for verification
                    )
                    new_task_params.append(theta_prime)
                task_params = new_task_params 

            
            # Validation rollouts (using head-only adapted params)
            worker_args = []
            for t, p in zip(tasks, task_params):
                p_cpu = {k: v.detach().cpu() for k, v in p.items()}
                worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                    policy_state_dict_cpu, p_cpu, self.batch_size, gamma))
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(ex.map(rollout_one_task_anil, worker_args))

            for task_idx, (mission, step_count, obs_list, action_list, reward_list, episode_list) in enumerate(results):
                valid_batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                for obs, action, reward, episode in zip(obs_list, action_list, reward_list, episode_list):
                    valid_batch.append([obs], [np.array(action)], [np.array(reward)], [np.array(episode)])
                self.baseline.fit(valid_batch)
                valid_batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)

                train_episodes_all.append(per_task_train_batches[task_idx])
                valid_episodes_all.append(valid_batch)
                all_step_counts.append(step_count)
            
        return (train_episodes_all, valid_episodes_all, all_step_counts)
