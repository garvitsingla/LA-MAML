This repository contains the code to reproduce results for **LA-MAML** (Language-adapted Model-Agnostic Meta-Learning).

---

## 1. Installation

Refer to the requirements.txt file to install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Evaluate Models & Reproduce Results (Direct Evaluation)

All the pre-trained model checkpoints are already included in this repository under their respective folders (`lang_model/`, `maml_model/`, `anil_model/`, and `lang_policy_model/`). Therefore, **direct evaluation can be done** to reproduce the results without any initial training.

*(The commands below use the `PickupDist` environment as a specific example, but evaluation can be executed on any of the trained environments: `GoToLocal`, `PickupDist`, `GoToObjDoor`, `GoToOpen`, `OpenDoor`, `OpenDoorLoc`, or `OpenDoorsOrder` by changing the `--env` parameter and its corresponding parameters).*

### 2.1 Main Evaluation

Evaluate and compare the performance of **LA-MAML** against the baselines (**MAML**, **ANIL**, and **Language-conditioned Policy**):

*(Note: The generated results will be logged and appended to `evaluation_results.xlsx`.)*

```bash
# Evaluate for PickupDist on a test configuration (e.g., room-size=8, num-dists=2)
python evaluation.py --env PickupDist --room-size 8 --num-dists 2 --max-steps 500 --delta-theta 0.3
```

### 2.2 Compare LA-MAML vs. MAML's Few-Shot Adaptation

Compare the pre-trained **LA-MAML** policy against standard **MAML** 2-step and 3-step adaptation baselines:

*(Note: The generated results will be logged and appended to `lamaml_maml_comparison_results.xlsx`.)*

```bash
# Compare LA-MAML with MAML 2-step and 3-step
python lamaml_maml_comparison.py --env PickupDist --room-size 8 --num-dists 2 --max-steps 500 --delta-theta 0.3
```

### 2.3 Test Ablation

Evaluate the ablation directly using the pre-trained models:

*(Note: The generated results will be logged and appended to `ablation_results.xlsx`.)*

```bash
python ablation.py --env PickupDist --room-size 8 --num-dists 2 --max-steps 500
```

---

## 3. Train Models from Scratch (Optional)

The models can be trained from scratch using the commands below.

*(Note: Due to variations in hardware configurations training models from scratch on different systems may produce different results.)*

*(The commands below use the `PickupDist` environment as an example.)*

**Train LA-MAML Model:**
```bash
python train_language.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500 --delta-theta 0.3
```

**Train Standard MAML Policy:**
```bash
python train_maml.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500
```

To run the comparison script (`lamaml_maml_comparison.py`) with models trained from scratch, the standard MAML baseline needs to be trained for 2 and 3 gradient steps by passing the `--num-steps` argument:

```bash
python train_maml.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500 --num-steps 2
python train_maml.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500 --num-steps 3
```

**Train ANIL Baseline:**
```bash
python train_anil.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500
```

**Train Language-Conditioned Policy:**
```bash
python train_language_conditioned_policy.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500
```
