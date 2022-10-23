import numpy as np
import time

class Rollout:
    def __init__(self, env, policy, experience_memory=None):
        self.env = env
        self.policy = policy
        self.experience_memory = experience_memory
    
    def run(self, args, rollout_mode='train'):
        """
        Run one episode.
        inputs:
            - args: arguments.
            - rollout_mode: 'train', 'test', # ! 'visualize'
        outputs:
            - rates: dict. rates of all different status.
            - times: dict.
                - list: time steps from episodes corresponding to different status.
            - average_times: dict. average time steps of all different status.
                - ['timeout']: use up step limit.
                - ['collision']: end in human-vehicle collision.
                - ['out-of-scene']: back up too much and exceed the backing-up distance limit.
                - ['success']: reached the goal within step limit and without collision.
        """
        # initialization
        self.policy.set_model_mode(model_mode='eval') # model_mode is evaluation during rollout, and training during policy update.
        # model_mode has nothing to do with the rollout_mode.
        if rollout_mode =='train':
            if not args.experience_replay:
                self.experience_memory.reset() # on-policy
        times = []
        accumulative_reward = []
        # multiple rollouts / parallel environments
        for _ in range(args.num_episodes_per_run):
            states = []
            actions = []
            rewards = []
            if args.policy_model == 'actor-critic':
                values = []
                action_log_probs = []
            else:
                raise RuntimeError("Policy model is not supported.")
            observation = self.env.reset()
            done = False
            n_steps = 0
            while not done:
                n_steps += 1
                if rollout_mode == 'visualize':
                    self.env.render()
                    time.sleep(0.01) # slow down the play speed in linux version
                # len: Tf. index: 0, 1, ..., Tf-1.
                # we use s_t, a_t, r_t. Different from David Silver's notation.
                # collect s_0, a_0, r_0, ..., s_t, a_t, r_t, ..., s_{Tf-1}, a_{Tf-1}, r_{Tf-1}.
                states.append(observation)
                if args.policy_model == 'actor-critic':
                    # collect ..., s_t, a_t, v_t, alp_t, r_t...
                    value, action, action_log_prob = self.policy.act(observation) # ! np -> np
                    actions.append(action)
                    values.append(value)
                    action_log_probs.append(action_log_prob)
                else:
                    # action = self.policy.act(observation) # np (5,)
                    raise RuntimeError("Policy model is not supported.")
                observation, reward, done, info = self.env.step(action)
                rewards.append(reward)

            if rollout_mode =='train':
                if args.policy_model == 'actor-critic':
                    # collect s_{Tf} and v_{Tf}.
                    states.append(observation)
                    value = self.policy.get_value(observation) # ! np -> np
                    values.append(value)
                    advantages, returns = self.compute_gae(values, rewards) # (Tf,) (Tf,)
                    experience = {}
                    # list (Tf,) for all experience
                    experience['states'] = states[:-1] 
                    experience['actions'] = actions
                    experience['rewards'] = rewards
                    experience['values'] = values[:-1]
                    experience['returns'] = returns
                    experience['advantages'] = advantages
                    experience['action_log_probs'] = action_log_probs
                    experience_length = n_steps # 0, 1, 2, ..., 999, info['step'] (if step_limit=1000)
                    for k in experience.keys():
                        assert type(experience[k]) == list and len(experience[k]) == experience_length
                    self.update_memory(args, experience)
                else:
                    raise RuntimeError("Policy model is not supported.")
            times.append(n_steps)
            accumulative_reward.append(sum(rewards))
        
        if rollout_mode =='train':
            self.experience_memory.update_tensor() # sync tensor with updated list in self.memory after rollouts.
        average_times = sum(times) / len(times)
        accumulative_reward = np.mean(accumulative_reward)
        return None, times, average_times, accumulative_reward


    def update_memory(self, args, experience):
        if args.experience_replay:
            self.experience_memory.push(experience)
        else:
            self.experience_memory.concatenate(experience)


    def compute_gae(self, values, rewards, gamma=0.99, lmbda=0.95, normalize_advantage=True):
        """
        Generalized Advantage Estimation
        inputs:
            - values # list: (Tf+1, ) # states (Tf+1, )
            - rewards # list: (Tf,)
        outputs:
            - advantages # list: (Tf,)
            - returns # list: (Tf,)
        """
        gae = 0.
        returns = []
        advantages = []
        for step in reversed(range(len(rewards))):
            # Tf-1, Tf-2, ..., 1, 0
            delta = rewards[step] + gamma * values[step + 1] - values[step] # first step is Tf-1
            gae = delta + gamma * lmbda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        if normalize_advantage:
            advantages = list((np.array(advantages)-np.mean(advantages))/(np.std(advantages)+1e-10))
        return advantages, returns