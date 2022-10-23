from os.path import join, exists

import gym
import torch

from arg_parse import arg_parse
from src.policy import Policy
from src.rollout import Rollout
from src.experience_memory import ExperienceMemory


def main():
    ##### Initialization Phase #####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    args = arg_parse()
    experience_memory = ExperienceMemory(args)
    env = gym.make('CartPole-v0')
    policy = Policy(args, device=device)
    rollout = Rollout(env, policy, experience_memory=experience_memory)
    if args.log is None:
        raise RuntimeError("The log for evaluation is None.")
    logdir = join('results', args.log)
    checkpoint_filepath = join(logdir, 'model_weights', str(args.training_episodes)+'.pt')
    if not exists(checkpoint_filepath):
        raise RuntimeError(checkpoint_filepath+" does not exist.")
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    policy.load_model_weights(checkpoint)
    print("model weights are loaded.")
    if args.visualize:
        rollout_mode = 'visualize'
    else:
        rollout_mode = 'test'
    _, _, _, accumulative_reward = rollout.run(args, rollout_mode=rollout_mode)
    print("accumulative reward: ", accumulative_reward)
    env.close()

if __name__ == "__main__":
    main()
