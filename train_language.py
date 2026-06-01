import os
import warnings
import logging
warnings.filterwarnings("ignore") 
logging.getLogger("gymnasium").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.multiprocessing as mp
from functools import partial
import numpy as np
import torch
import gc
import time
import os
import json
import matplotlib.pyplot as plt
import random
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.metalearners.lang_trpo import MAMLTRPO
import sampler_lang as S
from sampler_lang import (BabyAIMissionTaskWrapper, 
                        MissionEncoder,
                        SentenceMissionEncoder,
                        MissionParamAdapter, 
                        MultiTaskSampler, 
                        preprocess_obs)
from environment import (LOCAL_MISSIONS,
                         DOOR_MISSIONS,
                         OPEN_DOOR_MISSIONS,
                         DOOR_LOC_MISSIONS,
                         PICKUP_MISSIONS,
                         OPEN_DOORS_ORDER_MISSIONS)
from environment import (GoToLocalMissionEnv,
                         GoToOpenMissionEnv, 
                         GoToObjDoorMissionEnv,  
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenDoorsOrderMissionEnv)
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argparser
p = argparse.ArgumentParser()
p.add_argument("--env", dest="env_name",
               choices=["GoToLocal","PickupDist","GoToObjDoor","GoToOpen","OpenDoor",
                        "OpenDoorLoc","OpenDoorsOrder"],
               default="GoToLocal")
p.add_argument("--room-size", type=int, default=7)
p.add_argument("--num-dists", type=int, default=3)
p.add_argument("--max-steps", type=int, default=300)
p.add_argument("--delta-theta", type=float, default=0.3)
p.add_argument("--meta-iters", type=int, default=200, help="number of meta-batches")
p.add_argument("--batch-size", type=int, default=40, help="episodes per meta-batch (per task)")
p.add_argument("--num-workers", type=int, default=4)

args = p.parse_args()


# Build the environment
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
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions)


# Select for missions based on environment
def select_missions(env_name):
    mission_map = {
        "GoToLocal": LOCAL_MISSIONS,
        "PickupDist": PICKUP_MISSIONS,
        "GoToObjDoor": LOCAL_MISSIONS + DOOR_MISSIONS,
        "GoToOpen": LOCAL_MISSIONS,
        "OpenDoor": OPEN_DOOR_MISSIONS,
        "OpenDoorLoc": OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS,
        "OpenDoorsOrder": OPEN_DOORS_ORDER_MISSIONS,
    }
    return mission_map[env_name]


def main():


    def set_seed(seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed = 1
    set_seed(seed)

    env_name  = args.env_name
    room_size = args.room_size
    num_dists = args.num_dists
    max_steps = args.max_steps
    delta_theta = args.delta_theta
    num_workers = args.num_workers
    num_batches = args.meta_iters
    batch_size = args.batch_size


    missions = select_missions(env_name)

    make_env = partial(
        build_env,
        env_name,
        room_size,
        num_dists,
        max_steps,
        missions
    )

    env = make_env()
    print(f"Using environment: {env_name}\n"
          f"room_size: {room_size}  num_dists: {num_dists}  max_steps: {max_steps}  "
          f"delta_theta: {delta_theta}")

    # Policy setup 
    hidden_sizes = (64, 64)
    nonlinearity = torch.nn.functional.tanh

    mission_encoder = SentenceMissionEncoder(
        model_name="all-MiniLM-L6-v2",
        frozen=True,          
        normalize=True,         
        cache=True,           
        device=device
    )
    mission_encoder_output_dim = mission_encoder.output_dim
    mission_adapter_input_dimension = mission_encoder_output_dim

    # Policy Parameters shape
    obs, _ = env.reset()
    vec = preprocess_obs(obs)
    input_size = vec.shape[0]
    output_size = env.action_space.n


    policy = CategoricalMLPPolicy(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity,
    ).to(device)
    policy.share_memory()
    baseline = LinearFeatureBaseline(input_size).to(device)

    policy_param_shapes = [p.shape for p in policy.parameters()]

    mission_adapter_input_dimension = mission_encoder_output_dim
    mission_adapter = MissionParamAdapter(mission_adapter_input_dimension, policy_param_shapes).to(device)

    
    sampler = MultiTaskSampler(
        env=env,
        env_fn=make_env,
        batch_size=batch_size,     
        policy=policy,
        baseline=baseline,
        seed=1,
        num_workers=num_workers
    )

    meta_learner = MAMLTRPO(
        policy=policy,
        mission_encoder=mission_encoder,
        mission_adapter=mission_adapter,
        delta_theta=delta_theta,
        fast_lr=1e-4,
        first_order=True,
        device=device
    )

    # Training loop
    avg_steps_per_batch = []
    std_steps_per_batch = []
    meta_batch_size = globals().get("meta_batch_size") or min(5, len(env.missions))

    start_time = time.time()

    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")
        valid_episodes, step_counts = sampler.sample(
            meta_batch_size,
            meta_learner,
            gamma=0.99,
            gae_lambda=1.0,
            device=device
        )
        
        avg_steps = np.mean(step_counts) if len(step_counts) > 0 else float('nan')
        avg_steps_per_episode = avg_steps / sampler.batch_size 
        avg_steps_per_batch.append(avg_steps_per_episode)
        std_steps = np.std([s / sampler.batch_size for s in step_counts]) if len(step_counts) > 0 else 0.0
        std_steps_per_batch.append(std_steps)
        print(f"Average steps in Meta-batch {batch+1}: {avg_steps_per_episode}\n")

        meta_learner.step(valid_episodes,valid_episodes)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


    end_time = time.time()
    training_time = end_time - start_time
    time_per_iteration = training_time / num_batches
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Average time per iteration: {time_per_iteration:.2f} seconds")

    # Save the trained meta-policy parameters
    os.makedirs("lang_model", exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "mission_encoder": mission_encoder.state_dict(),
        "mission_adapter": mission_adapter.state_dict(),
    }, f"lang_model/lang_{env_name}_{delta_theta}.pth")


    # plot
    env_dir = os.path.join("metrics", env_name)
    os.makedirs(env_dir, exist_ok=True) 

    np.save(os.path.join(env_dir, f"la_maml_avg_steps_{delta_theta}.npy"), np.array(avg_steps_per_batch))
    np.save(os.path.join(env_dir, f"la_maml_std_steps_{delta_theta}.npy"), np.array(std_steps_per_batch))
    with open(os.path.join(env_dir, f"la_maml_meta_{delta_theta}.json"), "w") as f:
        json.dump({"label" : "LA-MAML", "env" : env_name}, f)
    
    plt.plot(avg_steps_per_batch)
    plt.xlabel("Meta-batch")
    plt.ylabel("Average steps per episode")
    plt.title(f"Average steps per episode per meta-batch (delta_theta={delta_theta})")
    plt.savefig(os.path.join(env_dir, f"la_maml_plot_{delta_theta}.png"))
    plt.close()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()