import argparse

def args_parse():
    parser = argparse.ArgumentParser(description="Atari: DQN")
    parser.add_argument('--ray', action="store_true", help='Train agent with given environment with ray')
    parser.add_argument('--play', help='Play agent with given environment')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parse()
    if args.ray:
        from PPO_RAY import Agent
        a = Agent()
        print("Start training with ray")
        a.train()
    else:
        from PPO import Agent
        a = Agent()
        if args.play:
            print("Start playing")
            a.play(args.play)
        else:
            print("Start training without ray")
            a.train()