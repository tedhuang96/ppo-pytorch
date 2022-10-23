import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader


class PPO:
    def __init__(self, policy, experience_memory, args):
        self.policy = policy
        self.data_loader = DataLoader(experience_memory, batch_size=args.batch_size, \
            shuffle=True, num_workers=4, drop_last=True) # drop_last=True for batchnorm in model.
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.clip_grad = args.clip_grad
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.optimizer = optim.Adam(self.policy.get_model_parameters(), lr=args.lr)
        self.device = self.policy.device

    def get_optimizer_state_dict(self):
        return self.optimizer.state_dict()

    def update(self):
        self.policy.set_model_mode(model_mode='train') # model_mode is evaluation during rollout, and training during policy update.
        value_loss_list, action_loss_list, dist_entropy_list = [], [], []
        for epoch in range(1, self.ppo_epoch+1):
            for batch_idx, batch in enumerate(self.data_loader):
                states_b, actions_b, rewards_b, values_b, returns_b, advantages_b, action_log_probs_b = batch
                states_b, actions_b, rewards_b, values_b, returns_b, advantages_b, action_log_probs_b = \
                    states_b.to(self.device), actions_b.to(self.device), rewards_b.to(self.device), \
                    values_b.to(self.device), returns_b.to(self.device), advantages_b.to(self.device), \
                    action_log_probs_b.to(self.device)
                
                values, action_log_probs, dist_entropy = self.policy.evaluate_actions(states_b, actions_b)
                ratio = torch.exp(action_log_probs - action_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantages_b
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = values_b + \
                        (values - values_b).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - returns_b).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - returns_b).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (values - returns_b).pow(2).mean()
                    
                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.get_model_parameters(), self.clip_grad)
                self.optimizer.step()

                value_loss_list.append(value_loss.item())
                action_loss_list.append(action_loss.item())
                dist_entropy_list.append(dist_entropy.item())
        
        value_loss_episode = np.mean(value_loss_list)
        action_loss_episode = np.mean(action_loss_list)
        dist_entropy_episode = np.mean(dist_entropy_list)
        return value_loss_episode, action_loss_episode, dist_entropy_episode