import torch.nn as nn

def make_mlp(dim_list, batchnorm=False, activation='relu', dropout=0.):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batchnorm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0.:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)