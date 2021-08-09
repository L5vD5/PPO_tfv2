# PPO_tfv2

## Requirements

- pybullet
- tensorflow v2.5
- ray
- gym
- tensorflow_probability

## Usage

```
python PPO.py       # without ray
python PPO_RAY.py   # with ray
```

## Config
You can change config.py to fit your own flag.

```
clip_ratio  # used for clipping output of log_std layer
pi_lr       # learning rate for policy parameter
vf_lr       # learning rate for value function parameter
epsilon     # TODO

gamma       # discout factor 
lam         # 


hdims       # dimension of hidden layers

# ray
n_cpu = n_workers # number of cpu

# Update
train_pi_iters
train_v_iters
target_kl
epochs
steps_per_epoch
ep_len_rollout
update_every
print_every
batch_size

# Evaluate
max_ep_len_eval
evaluate_every
```
