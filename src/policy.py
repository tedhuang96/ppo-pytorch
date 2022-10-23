import torch
from torch.distributions import Categorical

from src.model.actor_critic import ActorCritic


class Policy:
    def __init__(self, args, device='cuda:0'):
        if args.policy_model == 'actor-critic':
            self._model = ActorCritic(
                args.state_size,
                args.action_size,
                embedding_size=args.model_embedding_size,
                hidden_size=args.model_hidden_size,
                dropout=args.model_dropout,
            ).to(device)
        else:
            raise RuntimeError("Policy model is not supported.")
        self.device = device
    
    def load_model_weights(self, checkpoint):
        self._model.load_state_dict(checkpoint['model_state_dict'])

    def set_model_mode(self, model_mode='train'):
        r""" # ! model_mode is evaluation during rollout, and training during policy update."""
        if model_mode == 'train':
            self._model.train()
        elif model_mode == 'eval':
            self._model.eval()
        else:
            raise RuntimeError("Mode of the policy model is not supported.")
    
    def get_model_parameters(self):
        return self._model.parameters()
    
    def get_model_state_dict(self):
        return self._model.state_dict()

    def act(self, observation):
        """
        Policy acts in a rollout. Evaluation mode for the policy model.
        
        inputs:
            - observation: np # (state_size, )
        outputs:
            - value: scalar.
            - action: scalar.
            - action_log_prob: scalar.
        """
        with torch.no_grad():
            assert not self._model.training # evaluation mode
            x = torch.Tensor(observation).unsqueeze(0).to(self.device) # tensor (1, state_size)
            action_prob, state_value = self._model(x) # (1, action_size); (1, )
            action_prob, value = action_prob.squeeze(0), state_value.item() # (action_size, ); scalar
            m = Categorical(action_prob)
            action = m.sample() # tensor scalar
            action_log_prob = m.log_prob(action) # tensor scalar
            action, action_log_prob = action.item(), action_log_prob.item()
            return value, action, action_log_prob
        
    def get_value(self, observation):
        """
        Get value from the observation. 
        Typically called when retrieving the value of the terminal state in a rollout.
        Evaluation mode for the policy model.
        
        inputs:
            - observation: np # (state_size, )
        outputs:
            - value: scalar.
        """
        with torch.no_grad():
            assert not self._model.training # evaluation mode
            self._model.eval()
            x = torch.Tensor(observation).unsqueeze(0).to(self.device) # tensor (1, state_size)
            _, state_value = self._model(x) # _; (1, )
            value = state_value.item() # scalar
            return value

    def evaluate_actions(self, x, action):
        """
        Generate data for policy update via PPO. Training mode for the policy model.
        
        inputs:
            - x 
                # state data. 
                # tensor. size: (batch_size, state_size)
            - action
                # action data.
                # tensor. size: (batch_size, )
        outputs:
            - state_value
                # values of state from critic.
                # tensor. size: (batch_size, )
            - action_log_prob
                # log probability of the corresponding action data in rollouts
                # for the stochastic policy.
                # tensor. size: (batch_size,)
            - dist_entropy
                # entropy of action probability distribution
                # tensor scalar. size: (,)
        """
        assert self._model.training # training mode
        assert x.device == action.device == self.device # data on the same device
        action_prob, state_value = self._model(x) # (batch_size, action_size); (batch_size, )
        m = Categorical(action_prob)
        action_log_prob = m.log_prob(action) # (batch_size,)
        dist_entropy = m.entropy().mean() # scalar
        return state_value, action_log_prob, dist_entropy