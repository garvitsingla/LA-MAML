import torch
import numpy as np
import random
from functools import partial
from environment import (GoToLocalMissionEnv, 
                         GoToOpenMissionEnv, 
                         GoToObjDoorMissionEnv,  
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenDoorsOrderMissionEnv,
                         ActionObjDoorMissionEnv,
                         PutNextLocalMissionEnv, _aug_phrase)
from sampler_lang import BabyAIMissionTaskWrapper, MissionEncoder, MissionParamAdapter
import sampler_lang
import sampler_maml
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import pickle
import time
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJECTS = ['box']
COLORS = ['red', 'green', 'blue', 'purple','yellow', 'grey']
PREP_LOCS = ['on', 'at', 'to']

# Location names
LOC_NAMES = ['right', 'front']

DOOR_COLORS = ['yellow', 'grey']

# For Pickup
PICKUP_MISSIONS = [f"pick up the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For GoToLocal
LOCAL_MISSIONS = [f"go to the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For environments that include doors (GoToObjDoor, GoToOpen, Open)
DOOR_MISSIONS = [f"go to the {color} door" for color in DOOR_COLORS]
OPEN_DOOR_MISSIONS = [f"open the {color} door" for color in DOOR_COLORS]
DOOR_LOC_MISSIONS = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]
OPEN_TWO_DOORS_MISSIONS = [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
OPEN_DOORS_ORDER_MISSIONS = (
    [f"open the {c1} door" for c1 in DOOR_COLORS] +
    [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS] +
    [f"open the {c1} door after you open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
)

ACTION_OBJ_DOOR_MISSIONS = (
    [f"pick up the {c} box" for c in COLORS] +
    [f"go to the {c} box"   for c in COLORS] +
    [f"go to the {c} door"   for c in DOOR_COLORS] +
    [f"open a {c} door"      for c in DOOR_COLORS]
)


PUTNEXT_MISSIONS = [
    _aug_phrase(c1, t1, c2, t2)
    for c1 in COLORS for t1 in OBJECTS
    for c2 in COLORS for t2 in OBJECTS
    if not (c1 == c2 and t1 == t2)]


def build_env(env, room_size, num_dists, max_steps, missions):

    if env == "GoToLocal":
        base = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "PickupDist":
        base = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "GoToObjDoor":
        base = GoToObjDoorMissionEnv(max_steps=max_steps, num_distractors=num_dists)
    elif env == "GoToOpen":
        base = GoToOpenMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env == "OpenDoor":
        base = OpenDoorMissionEnv(room_size=room_size, max_steps=max_steps)
    elif env == "OpenDoorLoc":
        base = OpenDoorLocMissionEnv(room_size=room_size, max_steps=max_steps)
    elif env == "OpenDoorsOrder":
        base = OpenDoorsOrderMissionEnv(room_size=room_size)
    elif env == "ActionObjDoor":
        base = ActionObjDoorMissionEnv(objects=None, door_colors=None, obj_colors=None, num_dists=num_dists)
    elif env == "PutNextLocal":
        base = PutNextLocalMissionEnv(room_size=room_size, max_steps=max_steps, num_dists=None)
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions)


def select_missions(env_name):
    mission_map = {
        "GoToLocal": LOCAL_MISSIONS,
        "PickupDist": PICKUP_MISSIONS,
        "GoToObjDoor": LOCAL_MISSIONS + DOOR_MISSIONS,
        "GoToOpen": LOCAL_MISSIONS,
        "OpenDoor": OPEN_DOOR_MISSIONS,
        "OpenDoorLoc": OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS,
        "OpenDoorsOrder": OPEN_DOORS_ORDER_MISSIONS,
        "ActionObjDoor": ACTION_OBJ_DOOR_MISSIONS,
        "PutNextLocal": PUTNEXT_MISSIONS
    }
    return mission_map[env_name]


env_name = "PickupDist"
room_size = 7
num_dists = 2
max_steps = 350
delta_theta = 1

missions = select_missions(env_name)
make_env = partial(build_env, env_name, room_size, num_dists, max_steps, missions)
env = make_env()

print(f"env name {env}\n")


# restore saved lang-adapted policy 
ckpt = torch.load(f"lang_model/lang_policy_{env_name}.pth", map_location=device)
with open(f"lang_model/vectorizer_lang_{env_name}.pkl", "rb") as f:
    vectorizer = pickle.load(f)


sampler_lang.vectorizer = vectorizer  
mission_encoder_output_dim = 32
sampler_lang.mission_encoder = MissionEncoder(len(sampler_lang.vectorizer.get_feature_names_out()), 32, 64, mission_encoder_output_dim).to(device)
sampler_lang.mission_encoder.load_state_dict(ckpt["mission_encoder"])
sampler_lang.mission_encoder.eval()
mission_encoder = sampler_lang.mission_encoder
preprocess_obs = sampler_lang.preprocess_obs

dummy_obs, _ = env.reset()
input_size_lang = sampler_lang.preprocess_obs(dummy_obs).shape[0]
output_size = env.action_space.n
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# Policy language
policy_lang = CategoricalMLPPolicy(
    input_size=input_size_lang,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  
policy_lang.load_state_dict(ckpt["policy"])
policy_lang.eval()
policy_param_shapes = [p.shape for p in policy_lang.parameters()]

# Adapter
mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)
mission_adapter.load_state_dict(ckpt["mission_adapter"])    
mission_adapter.eval()


# restore saved maml policy

ckpt_base = f"maml_model/maml_{env_name}"
with open(ckpt_base + "_vectorizer.pkl", "rb") as f:
    sampler_maml.vectorizer = pickle.load(f)

# Policy maml
sampler_maml.mission_encoder = sampler_maml.MissionEncoder(
    len(sampler_maml.vectorizer.get_feature_names_out()),
    hidden_dim1=32, hidden_dim2=64, output_dim=32
).to(device)
sampler_maml.mission_encoder.load_state_dict(torch.load(ckpt_base + "_encoder.pth", map_location=device))
sampler_maml.mission_encoder.eval()

dummy_obs, _ = env.reset()
input_size_maml = sampler_maml.preprocess_obs(dummy_obs).shape[0]

policy_maml = CategoricalMLPPolicy(
        input_size=input_size_maml,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,      
    ).to(device)

# restore save maml policy
policy_maml.load_state_dict(torch.load(ckpt_base + ".pth", map_location=device))
policy_maml.eval()


baseline = LinearFeatureBaseline(input_size_maml).to(device)


def get_language_adapted_params(policy, mission_str, mission_encoder, mission_adapter, vectorizer, device):
    mission_vec = vectorizer.transform([mission_str]).toarray()[0]
    mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        mission_emb = mission_encoder(mission_tensor)
        mission_emb = mission_emb.to(device)
        delta_thetas = mission_adapter(mission_emb)
        delta_thetas = [delta * delta_theta  for delta in delta_thetas]
    policy_params = list(policy.parameters())
    param_names = list(dict(policy.named_parameters()).keys())
    from collections import OrderedDict
    theta_prime = OrderedDict(
        (name, param + delta.squeeze(0))
        for name, param, delta in zip(param_names, policy_params, delta_thetas)
    )

    return theta_prime


def adapt_policy_for_task(task, policy, num_steps=1, fast_lr=0.5, batch_size=10,baseline=None):
    
    env.reset_task(task)
    
    train_batches = []
    for _ in range(num_steps+1):
        batch = BatchEpisodes(batch_size=batch_size, gamma=0.99, device=device)
        for ep in range(batch_size):
            obs, info = env.reset()
            done = False
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            while not done:
                obs_vec = sampler_maml.preprocess_obs(obs)
                obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    dist = policy(obs_tensor)
                    action = dist.sample().item()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_obs.append(obs_vec)
                episode_actions.append(np.array(action))
                episode_rewards.append(np.array(reward, dtype=np.float32))
            batch.append(episode_obs, episode_actions, episode_rewards, [ep]*len(episode_obs))
        train_batches.append(batch)

    # Compute advantages 
    for batch in train_batches:
        batch.compute_advantages(baseline, gae_lambda=1.0, normalize=True)
    
    # Compute gradients and adapt policy parameters
    params = None
    for batch in train_batches:
        loss = reinforce_loss(policy, batch, params=params)
        params = policy.update_params(loss, params=params, step_size=fast_lr, first_order=True)
    return params



def evaluate_policy(env, policy,preprocess_obs=None, params=None, max_steps=max_steps, render=False):
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done and steps < max_steps:
        if render:
            env.render("human")
        obs_vec = preprocess_obs(obs)
        obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if params is not None:
                dist = policy(obs_tensor, params=params)
            else:
                dist = policy(obs_tensor)
            action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    return steps


# Evaluation
N_MISSIONS = 20
N_EPISODES = 40

results_lang = []
results_maml = []
results_random = []

print("Comparing language-adapted policy and random policy on random missions:")
for i in range(N_MISSIONS):
    mission = random.choice(missions)
    # print(f"\nMission {i+1}/{N_MISSIONS}: '{mission}'")

    # 1. Lang-adapted policy
    # print("  [Lang-adapted policy episodes]")
    theta_prime = get_language_adapted_params(policy_lang, mission, mission_encoder, mission_adapter, vectorizer, device)
    lang_steps = []

    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_lang, preprocess_obs= sampler_lang.preprocess_obs, params=theta_prime)
        lang_steps.append(steps)
    mean_lang = np.mean(lang_steps)
    std_lang = np.std(lang_steps)
    results_lang.append(mean_lang)
    # print(f"    --> Avg steps: {mean_lang:.2f} ± {std_lang:.2f}")


    # 2. MAML policy
    # print(" [Evaluating with maml adaptation]")
    maml_params = adapt_policy_for_task(mission, policy_maml, num_steps=2, fast_lr=0.25, batch_size=10, baseline=baseline)
    maml_steps = []

    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_maml, preprocess_obs= sampler_maml.preprocess_obs, params=maml_params)
        maml_steps.append(steps)
    mean_maml = np.mean(maml_steps)
    std_maml = np.std(maml_steps)
    results_maml.append(mean_maml)
    # print(f"    --> Avg steps: {mean_maml:.2f} ± {std_maml:.2f}")


    # 3. Randomly initialized policy
    # print("  [Random policy episodes]")
    scratch_policy = CategoricalMLPPolicy(
        input_size=input_size_lang,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,
    ).to(device)
    scratch_policy.eval()
    rand_steps = []
    
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, scratch_policy, preprocess_obs=sampler_lang.preprocess_obs)
        rand_steps.append(steps)
    mean_rand = np.mean(rand_steps)
    std_rand = np.std(rand_steps)
    results_random.append(mean_rand)
    # print(f"    --> Avg steps: {mean_rand:.2f} ± {std_rand:.2f}")

end_time = time.time()

print(f"Execution time: {(end_time - start_time)/60} minutes\n")

# Results
print("\n===== FINAL AGGREGATE RESULTS =====")
print(f"Lang-adapted policy: {np.mean(results_lang):.2f} ± {np.std(results_lang):.2f}")
print(f"MAML policy:   {np.mean(results_maml):.2f} ± {np.std(results_maml):.2f}")
print(f"Random initializations:  {np.mean(results_random):.2f} ± {np.std(results_random):.2f}")
