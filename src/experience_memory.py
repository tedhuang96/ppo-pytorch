import torch
from torch.utils.data import Dataset

class ExperienceMemory(Dataset):
    """
    reference: https://github.com/vita-epfl/CrowdNav/blob/503173b836d5460e30234df7e14a7c67ee0ebfc7/crowd_nav/utils/memory.py#L4
    ExperienceMemory can be used to store rollouts for policy update.
    """
    def __init__(self, args):
        self.args = args
        self.capacity = self.args.memory_capacity
        self.reset()
    

    def __len__(self):
        for k in self.memory.keys():
            assert self.length == len(self.memory[k])
        return self.length
    

    def push(self, experience):
        """push the newest in and the oldest out. works in experience replay setting."""
        assert self.memory.keys() == experience.keys()
        experience_length = len(list(experience.values())[0])
        for i in range(experience_length):
            if self.length < self.position + 1: # before capacity is filled.
                self.length += 1
                for k in self.memory.keys():
                    self.memory[k].append(experience[k][i])
            else: # after capacity is filled, replace old experience with newer ones.
                for k in self.memory.keys():
                    self.memory[k][self.position] = experience[k][i]
            self.position = (self.position + 1) % self.capacity
    

    def concatenate(self, experience):
        """concatenate with new memory (list). This means we don't do experience replay.
        If we do concatenation, we must assure the concatenated won't exceed the capacity."""
        assert self.memory.keys() == experience.keys()
        experience_length = len(list(experience.values())[0])
        if self.length + experience_length > self.capacity:
            raise RuntimeError("Adding new experience exceeds the memory capacity.")
        for k in self.memory.keys():
            self.memory[k] = self.memory[k] + experience[k]
        self.length += experience_length
        self.position += experience_length # sync of self.position with self.length when not reach capacity.

    def is_full(self):
        return self.length == self.capacity
    

    def reset(self):
        self.memory = {}
        self.position = 0
        self.length = 0
        if self.args.policy_model == 'actor-critic':
            self.memory_keys = ['states', 'actions', 'rewards', 'values', 'returns', \
                'advantages', 'action_log_probs'] # ordered keys
            for k in self.memory_keys:
                self.memory[k] = []
        else:
            raise RuntimeError("Policy model is not supported.")
        self.update_tensor()
    
    def update_tensor(self):
        """
        list2tensor.
        Called in reset to create empty tensors, or
        Called after self.memory (which is a dict of list) is updated from rollouts.
        outputs:
            - memory_tensor: dict of tensors.
                - ['states']: (data_len, 5)
                - ['actions']: (data_len,)
                - ['rewards']: (data_len,)
                - ['values']: (data_len,)
                - ['returns']: (data_len,)
                - ['advantages']: (data_len,)
                - ['action_log_probs']: (data_len,)
        """
        self.memory_tensor = {}
        for k in self.memory.keys():
            self.memory_tensor[k] = torch.Tensor(self.memory[k]) # float32
    
    
    def __getitem__(self, index):
        """
        if self.args.policy_model == 'actor-critic':
            return a list in the following order:
                - states[index]
                - actions[index]
                - rewards[index]
                - values[index]
                - returns[index]
                - advantages[index]
                - action_log_probs[index] # all are not .to(device) yet
        """
        out = []
        for k in self.memory_keys: # ordered keys
            out.append(self.memory_tensor[k][index])
        return out