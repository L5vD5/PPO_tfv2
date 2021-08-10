# PPO_tfv2

## Requirements

- pybullet
- tensorflow v2.5
- ray
- gym
- tensorflow_probability

## Usage

```
$ python main.py --hdims 128 128                            # train without ray
$ python main.py --ray [number of worker] --hdims 128 128   # train with ray
$ python main.py --play [weight_file_path] --hdims 128 128  # play with weight file

$ python main.py --help

usage: main.py [-h] [--ray RAY] [--play PLAY] --hdims HDIMS [HDIMS ...]
               [--clip_ratio CLIP_RATIO] [--pi_lr PI_LR] [--vf_lr VF_LR]
               [--gamma GAMMA] [--lam LAM] [--train_pi_iters TRAIN_PI_ITERS]
               [--train_v_iters TRAIN_V_ITERS] [--target_kl TARGET_KL]
               [--epochs EPOCHS] [--max_ep_len MAX_EP_LEN]
               [--steps_per_epoch STEPS_PER_EPOCH] [--print_every PRINT_EVERY]
               [--ep_len_rollout EP_LEN_ROLLOUT] [--batch_size BATCH_SIZE]
               [--evaluate_every EVALUATE_EVERY]

Atari: DQN

optional arguments:
  -h, --help            show this help message and exit
  --ray RAY             Train agent with given environment with ray
  --play PLAY           Play agent with given environment
  --hdims HDIMS [HDIMS ...]
                        size of hidden dimension
  --clip_ratio CLIP_RATIO
                        Hyperparameter for clipping in the policy objective
  --pi_lr PI_LR         learning rate for policy parameter
  --vf_lr VF_LR         learning rate for value function parameter
  --gamma GAMMA         discount factor
  --lam LAM             Lambda for GAE-Lambda
  --train_pi_iters TRAIN_PI_ITERS
                        Maximum number of gradient descent steps to take on
                        policy loss per epoch
  --train_v_iters TRAIN_V_ITERS
                        Number of gradient descent steps to take on value
                        function per epoch
  --target_kl TARGET_KL
                        Roughly what KL divergence we think is appropriate
                        between new and old policies after an update. This
                        will get used for early stopping
  --epochs EPOCHS       Number of epochs of interaction (equivalent to number
                        of policy updates) to perform
  --max_ep_len MAX_EP_LEN
                        Maximum length of trajectory
  --steps_per_epoch STEPS_PER_EPOCH
                        How many steps per epoch
  --print_every PRINT_EVERY
                        How often to print result
  --ep_len_rollout EP_LEN_ROLLOUT
                        How many episodes per rollout in ray
  --batch_size BATCH_SIZE
                        How big batch size in ray
  --evaluate_every EVALUATE_EVERY
                        How often to evaluate agent
```
