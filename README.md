
This repository contains the code to reproduce results in supplementary material for **LA-MAML** (Language-adapted Model-Agnostic Meta-Learning).

## 1. Installation

Refer to the requirements.txt file to install dependencies.

## 2. Train Models

> **Note:** We do not provide any pre-trained models due to space constraints. In order to reproduce the results, **explicit training is required for all the models** before running the evaluation scripts.

*(The commands below use the `PickupDist` environment as a specific example, but models can be trained on **any of the environments as mentioned in the paper** by changing the `--env` parameter and its other parameters.)*

**Train LA-MAML Model:**
```bash
python train_language.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500 --delta-theta 0.3
```

**Train Standard MAML Policy:**
```bash
python train_maml.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500
```

**Train ANIL Baseline:**
```bash
python train_anil.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500
```

**Train Language-Conditioned Policy:**
```bash
python train_language_conditioned_policy.py --env PickupDist --room-size 7 --num-dists 2 --max-steps 500
```

---

## 3. Evaluate Models & Reproduce Results

Once all the respective models are completely trained, use the scripts below to evaluate them.

### 3.1 Main Evaluation

*(Note: The generated results will be logged and appended to `evaluation_results.xlsx`.)*

```bash
# Evaluate for PickupDist on any new configuration, say room-size=8, num-dists=2
python evaluation.py --env PickupDist --room-size 8 --num-dists 2 --max-steps 500 --delta-theta 0.3
```

### 3.2 Compare LA-MAML vs. MAML's few-shot adaptation

> **Note:** For this comparison script to work, an additional training is required for standard MAML baseline for 2 and 3 gradient steps (using the `--num-steps 2` and `--num-steps 3` flags during the training phase in Section 2).

To compare the trained LA-MAML policy against the standard MAML 2-step and 3-step baselines, run the comparison script:

*(Note: The generated results will be logged and appended to `lamaml_maml_comparison_results.xlsx`.)*

**Example Command:**
```bash
python lamaml_maml_comparison.py --env PickupDist --room-size 8 --num-dists 2 --max-steps 500 --delta-theta 0.3
```

### 3.3 Test Ablation

Evaluate the ablation using the following testing script.

*(Note: The generated results will be logged and appended to `ablation_results.xlsx`.)*

```bash
python ablation.py --env PickupDist --room-size 8 --num-dists 2 --max-steps 500
```
