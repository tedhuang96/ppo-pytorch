
import datetime
from os import mkdir
from os.path import join, isdir

import gym
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.ppo import PPO
from src.policy import Policy
from src.rollout import Rollout
from src.experience_memory import ExperienceMemory
from arg_parse import arg_parse


def main():
    ##### Initialization Phase #####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    args = arg_parse()
    print(args)
    experience_memory = ExperienceMemory(args)
    env = gym.make('CartPole-v0')
    policy = Policy(args, device=device)
    rollout = Rollout(env, policy, experience_memory=experience_memory)
    ppo_agent = None
    # initialize tensorboard writer
    writername = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = join('results', writername)
    writer = SummaryWriter(logdir)
    print('Started writing in folder: '+logdir)
    ##### Training Phase #####
    episodes = 0
    pbar = tqdm(total=args.training_episodes) 
    while episodes < args.training_episodes:
        _, _, average_times, accumulative_reward = rollout.run(args, rollout_mode='train')
        if ppo_agent is None:
            ppo_agent = PPO(policy, experience_memory, args)
        value_loss, action_loss, dist_entropy = ppo_agent.update()
        episodes += args.num_episodes_per_run
        pbar.update(args.num_episodes_per_run)
        writer.add_scalar('accumulative reward', accumulative_reward, episodes)
        writer.add_scalar('value loss', value_loss, episodes)
        writer.add_scalar('action loss', action_loss, episodes)
        writer.add_scalar('distribution entropy', dist_entropy, episodes)
        writer.add_scalar('average time', average_times, episodes)
        if episodes % args.num_episodes_per_checkpoint == 0:
            modeldir = join(logdir, 'model_weights')
            if not isdir(modeldir):
                mkdir(modeldir)    
            model_filename = join(modeldir, str(episodes)+'.pt')
            torch.save({
                'episodes': episodes,
                'model_state_dict': policy.get_model_state_dict(),
                'optimizer_state_dict': ppo_agent.get_optimizer_state_dict(),
                }, model_filename)
            print(model_filename+' is saved.')
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
