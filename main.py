import argparse

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--ray', type=int, help='Train agent with given environment with ray')
    parser.add_argument('--play', help='Play agent with given environment')
    parser.add_argument('--hdims', type=int, nargs='+', help='size of hidden dimension', required=True)
    parser.add_argument('--clip_ratio', default=0.2, help='Hyperparameter for clipping in the policy objective')
    parser.add_argument('--pi_lr', default=3e-4, help='learning rate for policy parameter')
    parser.add_argument('--vf_lr', default=1e-3, help='learning rate for value function parameter')
    parser.add_argument('--gamma', default=0.99, help='discount factor')
    parser.add_argument('--lam', default=0.95, help='Lambda for GAE-Lambda')
    parser.add_argument('--train_pi_iters', default=100, help='Maximum number of gradient descent steps to take on policy loss per epoch')
    parser.add_argument('--train_v_iters', default=100, help='Number of gradient descent steps to take on value function per epoch')
    parser.add_argument('--target_kl', default=0.01, help='Roughly what KL divergence we think is appropriate between new and old policies after an update. This will get used for early stopping')
    parser.add_argument('--epochs', default=1000, help='Number of epochs of interaction (equivalent to number of policy updates) to perform')
    parser.add_argument('--max_ep_len', default=1000, help='Maximum length of trajectory')
    parser.add_argument('--steps_per_epoch', default=5000, help='How many steps per epoch')
    parser.add_argument('--print_every', default=10, help='How often to print result')
    parser.add_argument('--ep_len_rollout', default=500, help='How many episodes per rollout in ray')
    parser.add_argument('--batch_size', default=4096, help='How big batch size in ray')
    parser.add_argument('--evaluate_every', default=50,  help='How often to evaluate agent')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()
    if args.ray:
        from PPO_RAY import Agent
        a = Agent(args)
        print("Start training with ray")
        a.train()
    else:
        from PPO import Agent
        a = Agent(args)
        if args.play:
            print("Start playing")
            a.play(args.play)
        else:
            print("Start training without ray")
            a.train()