import torch.nn as nn
import torch.nn.functional as F

from src.model.utils import make_mlp


class ActorCritic(nn.Module):
    """Actor Critic."""
    def __init__(self, state_size, action_size, \
        embedding_size=64, hidden_size=128, dropout=0.1):
        super(ActorCritic, self).__init__()
        # spatial embeddings
        self.actor_embedding = make_mlp([state_size, hidden_size, embedding_size], \
            batchnorm=True, activation='relu', dropout=dropout)
        self.critic_embedding = make_mlp([state_size, hidden_size, embedding_size], \
            batchnorm=True, activation='relu', dropout=dropout)
        # encoder
        self.actor_encoder = make_mlp([embedding_size, hidden_size], \
            batchnorm=True, activation='relu', dropout=dropout)
        self.critic_encoder = make_mlp([embedding_size, hidden_size], \
            batchnorm=True, activation='relu', dropout=dropout)
        self.actor_fc = nn.Linear(hidden_size, action_size)        
        self.critic_fc = nn.Linear(hidden_size, 1)

    
    def forward(self, x):
        """
        inputs:
            - x 
                # input data. 
                # tensor. size: (batch_size, state_size)
        outputs:
            - action_prob
                # probability of actions in the stochastic policy.
                # tensor. size: (batch_size, action_size)
            - state_value
                # values of state from critic.
                # tensor. size: (batch_size, )
        """
        xa = self.actor_embedding(x)
        action_prob = F.softmax(self.actor_fc(self.actor_encoder(xa)), dim=-1) # (batch_size, action_size)
        xc = self.critic_embedding(x)
        state_value = self.critic_fc(self.critic_encoder(xc)).squeeze(-1) # (batch_size, )
        return action_prob, state_value