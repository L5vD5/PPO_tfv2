# PPO_tfv2

## Requirements

- pybullet
- tensorflow v2.5
- ray
- gym
- tensorflow_probability

## Usage

```
$ python main.py --help

$ python main.py                            # train without ray
$ python main.py --ray                      # train with ray
$ python main.py --play [weight_file_path]  # play with weight file

usage: main.py [-h] [--ray] [--play PLAY]

Atari: DQN

optional arguments:
  -h, --help   show this help message and exit
  --ray        Train agent with given environment with ray
  --play PLAY  Play agent with given environment
```

## Config
You can change config.py to fit your own flag.

```
clip_ratio  # used for clipping output of log_std layer
pi_lr       # learning rate for policy parameter
vf_lr       # learning rate for value function parameter
epsilon     # TODO

gamma       # discount factor 
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
