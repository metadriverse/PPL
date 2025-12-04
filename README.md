# Predictive Preference Learning from Human Interventions (PPL)

<h3><b>NeurIPS 2025 Spotlight</b></h3>

Official release for the code used in the paper: *Predictive Preference Learning from Human Interventions*

[**Webpage**](https://metadriverse.github.io/ppl/) | 
[**Poster**](https://github.com/metadriverse/PPL/blob/main/PPL_Poster.pdf) |
[**Paper**](https://arxiv.org/pdf/2510.01545) |
[**Demo**](https://www.youtube.com/watch?v=Bw9-0g3F1Tg)

[![](https://github.com/metadriverse/PPL/blob/main/TestSR.png)](https://metadriverse.github.io/ppl/)

## Installation

```bash
git clone https://github.com/metadriverse/PPL.git
cd PPL

# Create Conda environment
conda create -n ppl python=3.7
conda activate ppl

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Launch Experiments

### Predictive Preferenece Learning (Ours)

To reproduce the main experiment reported in the paper, run the training script `train_ppl_metadrive.py` in the folder `ppl/experiments/metadrive`. It takes about 12 minutes to train a performant driving agent. We also provide a simpler toy environment with `--toy_env`.

```bash
cd ~/PPL

# Run toy experiment
python ppl/experiments/metadrive/train_ppl_metadrive.py \
--toy_env

# Run full experiment
python ppl/experiments/metadrive/train_ppl_metadrive.py \
--wandb \
--wandb_project WADNB_PROJECT_NAME \
--wandb_team WANDB_ENTITY_NAME \
```

You can specify the output length H of the trajectory predictor with `--num_predicted_steps H`. You can also set different preference horizons L with `--preference_horizon L`.

To train a neural expert approximating human policy, you can run the following command:

```bash
# Train PPO expert (Optional)
python ppl/experiments/metadrive/train_ppo_metadrive.py
```

### Baselines

We also provide the codes for the baselines and ablation studies. 

For example, to run the baseline Proxy Value Propagation (Peng et al. 2023), you can run the following command:

```bash
# Run Proxy Value Propagation (Baseline)
python ppl/experiments/metadrive/train_pvp_metadrive.py
```

You can also set the mode --only_bc_loss=True in our PPL method to verify that the DPO-like preference loss contributes to improving the training performance of PPL.

```bash
# PPL without Preference Loss (Baseline)
python ppl/experiments/metadrive/train_ppl_metadrive.py \
--only_bc_loss=True
```
