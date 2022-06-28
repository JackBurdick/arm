from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_units: List[int] = None,
        seed: int = 42,
    ):
        """Create and initialize model layers

        Parameters
        ----------
        state_size : int
            size of the state vector
        action_size : int
            size of the action vector
        hidden_units : List[int], optional
            List of number of nodes per layer, by default None
        seed : int
            random seed
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)

        # input and hidden
        units = [state_size, *hidden_units]
        layer_list = []
        for i, u in enumerate(units):
            if i != 0:
                layer_list.append(nn.Linear(units[i - 1], u))
        self.hidden_layers = nn.ModuleList(layer_list)

        # output
        self.out_layer = nn.Linear(units[-1], action_size)

        # initialize layers
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.bn(state)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.out_layer(x)
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_1: int = 256,
        fc_2: int = 256,
        fc_3: int = 256,
        seed: int = 42,
    ):
        """Create and initialize model layers

        Parameters
        ----------
        state_size : int
            size of the state vector
        action_size : int
            size of the action vector
        fc_1 : int, optional
            number of nodes in the first dense layer, by default 256
        fc_2 : int, optional
            number of nodes in the second dense layer, by default 256
        fc_3 : int, optional
            number of nodes in the third dense layer, by default 256
        seed : int, optional
            random seed, by default 42
        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)

        # input and hidden
        self.fc_1 = nn.Linear(state_size, fc_1)
        self.fc_2 = nn.Linear(fc_1 + action_size, fc_2)
        self.fc_3 = nn.Linear(fc_2, fc_3)

        # output
        self.fc_out = nn.Linear(fc_3, 1)

        # initialize layers
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.fc_3.weight.data.uniform_(*hidden_init(self.fc_3))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.bn(state)
        x = F.relu(self.fc_1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        out = self.fc_out(x)
        return out
