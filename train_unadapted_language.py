import torch.multiprocessing as mp
from functools import partial
import numpy as np
import torch
import matplotlib.pyplot as plt 
import pickle
import time
import gc
import os
import json
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
from maml_rl.metalearners.maml_trpo3 import MAMLTRPO
from sklearn.feature_extraction.text import CountVectorizer
import sampler_lang as S
from sampler_lang import (BabyAIMissionTaskWrapper, 
                        MissionEncoder, 
                        MissionParamAdapter, 
                        MultiTaskSampler, 
                        preprocess_obs)
from environment import (LOCAL_MISSIONS, LOCAL_MISSIONS_VOCAB,
                        DOOR_MISSIONS, DOOR_MISSIONS_VOCAB,
                        OPEN_DOOR_MISSIONS, OPEN_DOOR_MISSIONS_VOCAB,
                        DOOR_LOC_MISSIONS,  DOOR_LOC_MISSIONS_VOCAB,
                        PICKUP_MISSIONS, PICKUP_MISSIONS_VOCAB,
                        OPEN_DOORS_ORDER_MISSIONS, OPEN_DOORS_ORDER_MISSIONS_VOCAB,
                        ACTION_OBJ_DOOR_MISSIONS, ACTION_OBJ_DOOR_MISSIONS_VOCAB,
                        PUTNEXT_MISSIONS, PUTNEXT_MISSIONS_VOCAB)
                        
from environment import (GoToLocalMissionEnv, 
                            GoToOpenMissionEnv, 
                            GoToObjDoorMissionEnv, 
                            PickupDistMissionEnv,
                            OpenDoorMissionEnv,
                            OpenDoorLocMissionEnv,
                            OpenDoorsOrderMissionEnv,
                            ActionObjDoorMissionEnv,
                            PutNextLocalMissionEnv)


start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    elif env == "ActionObjDoor":
        base = ActionObjDoorMissionEnv(objects=None, door_colors=None, obj_colors=None, num_dists=num_dists)
    elif env == "PutNextLocal":
        base = PutNextLocalMissionEnv(room_size=room_size, max_steps=max_steps, num_dists=None)
    else:
        raise ValueError(f"Unknown env_name: {env}")

    return BabyAIMissionTaskWrapper(base, missions=missions)


# Select for missions & vocab based on environment
def select_missions_and_vocab(env):

    if env == "GoToLocal":
        return LOCAL_MISSIONS, LOCAL_MISSIONS_VOCAB
    if env == "PickupDist":
        return PICKUP_MISSIONS, PICKUP_MISSIONS_VOCAB
    if env == "GoToObjDoor":
        return (LOCAL_MISSIONS + DOOR_MISSIONS), (LOCAL_MISSIONS_VOCAB + DOOR_MISSIONS_VOCAB)
    if env == "GoToOpen":
        return LOCAL_MISSIONS, LOCAL_MISSIONS_VOCAB
    if env == "OpenDoor":
        return OPEN_DOOR_MISSIONS, OPEN_DOOR_MISSIONS_VOCAB
    if env == "OpenDoorLoc":
        return (OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS), (OPEN_DOOR_MISSIONS_VOCAB + DOOR_LOC_MISSIONS_VOCAB)
    if env == "OpenDoorsOrder":
        return OPEN_DOORS_ORDER_MISSIONS, OPEN_DOORS_ORDER_MISSIONS_VOCAB
    if env == "ActionObjDoor":
        return ACTION_OBJ_DOOR_MISSIONS, ACTION_OBJ_DOOR_MISSIONS_VOCAB
    if env == "PutNextLocal":
        return PUTNEXT_MISSIONS, PUTNEXT_MISSIONS_VOCAB 
    raise ValueError(f"Unknown env for missions/vocab: {env}")



def main():

    env_name = "GoToOpen"
    room_size=7
    num_dists=10
    max_steps=600
    delta_theta=0.7
    num_workers=4
    num_batches=50

    missions, vocabs = select_missions_and_vocab(env_name)
    vectorizer = CountVectorizer(ngram_range=(1, 2), lowercase=True)
    vectorizer.fit(vocabs)

    make_env = partial(
        build_env,
        env_name,
        room_size,
        num_dists,
        max_steps,
        missions
    )

    env = make_env()
    model = env_name
    print(f"Using environment: {env_name}\n"
          f"room_size: {room_size}  num_dists: {num_dists}  max_steps: {max_steps}  "
          f"delta_theta: {delta_theta}")

    # Policy setup 
    hidden_sizes = (64, 64)
    nonlinearity = torch.nn.functional.tanh

    # Instantiate the encoder 
    S.vectorizer = vectorizer
    mission_encoder_input_dim = len(S.vectorizer.get_feature_names_out())
    mission_encoder_output_dim = 32  
    mission_encoder = MissionEncoder(mission_encoder_input_dim,  hidden_dim1=32, hidden_dim2=64, output_dim=mission_encoder_output_dim).to(device)
    S.mission_encoder = mission_encoder  

    # Finding Policy Parameters shape
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

    
    # Sampler setup
    sampler = MultiTaskSampler(
        env=env,
        env_fn=make_env,
        batch_size=50,     
        policy=policy,
        baseline=baseline,
        seed=1,
        num_workers=num_workers
    )

    # Meta-learner setup
    meta_learner = MAMLTRPO(
        policy=policy,
        mission_encoder=mission_encoder,
        mission_adapter=mission_adapter,
        vectorizer=vectorizer,
        delta_theta=delta_theta,
        fast_lr=1e-4,
        first_order=True,
        device=device
    )

    # Training loop
    avg_steps_per_batch = []
    meta_batch_size = globals().get("meta_batch_size") or min(5, len(env.missions))

    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")
        valid_episodes, step_counts = sampler.sample(
            meta_batch_size,
            meta_learner,
            num_steps=1,
            fast_lr=1e-4,
            gamma=0.99,
            gae_lambda=1.0,
            device=device
        )
        
        avg_steps = np.mean(step_counts) if len(step_counts) > 0 else float('nan')
        avg_steps_per_episode = avg_steps / sampler.batch_size 
        avg_steps_per_batch.append(avg_steps_per_episode)
        print(f"Average steps in Meta-batch {batch+1}: {avg_steps_per_episode}\n")

        meta_learner.step(valid_episodes,valid_episodes)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    end_time = time.time()

    print(f"Execution time: {(end_time - start_time)/60}minutes")


    # # Save the trained meta-policy parameters
    # torch.save({
    #     "policy": policy.state_dict(),
    #     "mission_encoder": mission_encoder.state_dict(),
    #     "mission_adapter": mission_adapter.state_dict()
    # }, f"ablation_model/lang_policy_{model}.pth")

    # # Save the vectorizer
    # with open(f"ablation_model/vectorizer_lang_{model}.pkl", "wb") as f:
    #     pickle.dump(vectorizer, f)

    # print("Meta-training for unadapted lang policy finished")


    # # Save metrics
    # env_dir = os.path.join("metrics", env_name)
    # os.makedirs(env_dir, exist_ok=True)

    # np.save(os.path.join(env_dir, "ablation_avg_steps.npy"), np.array(avg_steps_per_batch))
    # with open(os.path.join(env_dir, "ablation_meta.json"), "w") as f:
    #     json.dump({"label": "ablation", "env": env_name}, f)


    # # After training, plot    
    # plt.plot(avg_steps_per_batch)
    # plt.xlabel("meta-iteration")
    # plt.ylabel("Average steps")
    # plt.title(f"unadapted_language_{model}_{room_size}_{num_dists}_{max_steps}_{delta_theta}_{num_batches}")
    # plt.show()

    pass

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()