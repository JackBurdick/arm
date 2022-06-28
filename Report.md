[//]: # (Image References)

[training_plot]: ./assets/training_plot.png "training_plot"
[smoothed_training_plot]: ./assets/smoothed_training_plot.png "training_plot"


# Arm (Unity reacher environment)

## Context

The goal environment information can be found in the main [readme.md](./README.md)


## Description

Below is an overview of:
 - loss plots
 - model
 - agent
 - config settings

### Loss Plots

The raw loss plot (below) shows a nice/rapid rise in performance, followed by a dip and maybe a slight decrease in performance.

![Reward over time][training_plot]

The smoothed version (below) is unnecessary, but does confirm the suspected slight decrease in performance follwing the sharp rise in performance.

![Smoothed reward over time][smoothed_training_plot]

### Model (Actor/Critic)

The models aren't terribly exciting, largely standard DNNs. Interestingly, I
found that more complicated architectures or even slight changes to these
architectures had surprising effects on the model performance. For example
changing the hidden activations from `relu` to `elu` caused the model to train
much slower. Attempting more complicated architectures, like a transformer w/a
learned positional embedding, lead to poor performance and didn't seem to any
meaningful training.

#### Actor

The core actor model consists of a number of linear layers of pre-specified units in
this case `[512, 256, 128, 64]`. Narrower models (e.g. starting with 320 nodes
in the first layer), even with additional layers subjectively performed worse.

```python
self.bn = nn.BatchNorm1d(state_size)

# input and hidden
units = [state_size, *hidden_units]
layer_list = []
for i, u in enumerate(units):
    if i != 0:  # skip first
        layer_list.append(nn.Linear(units[i - 1], u))
self.hidden_layers = nn.ModuleList(layer_list)

self.out_layer = nn.Linear(units[-1], action_size)
```

Followed by corresponding branches for the advantage and state:

```python
def forward(self, state):
    x = self.bn(state)
    for layer in self.hidden_layers:
        x = F.relu(layer(x))
    x = self.out_layer(x)
    return torch.tanh(x)
```


#### Critic

The critic model was similar to the actor. With layers of size: `fc_1: 384,
fc_2: 256, fc_3: 128`.

```python
self.bn = nn.BatchNorm1d(state_size)

# input and hidden
self.fc_1 = nn.Linear(state_size, fc_1)
self.fc_2 = nn.Linear(fc_1 + action_size, fc_2)
self.fc_3 = nn.Linear(fc_2, fc_3)

# output
self.fc_out = nn.Linear(fc_3, 1)
```

The difference, is that in this case we have multiple inputs (state and action)
and so we concatenated a projection of the state to the action before using the
rest of the architecture. I think this design was a hold over from a prior
exercise and seemed to work better than other design choices

```python
def forward(self, state, action):
    state = self.bn(state)
    x = F.relu(self.fc_1(state))
    x = torch.cat((x, action), dim=1)
    x = F.relu(self.fc_2(x))
    x = F.relu(self.fc_3(x))
    out = self.fc_out(x)
    return out
```

### Agent

The main algorithm was a `DDPG` ([Deep deterministic Policy Gradient](https://arxiv.org/abs/1509.02971v6)). Some overview blogs can be found [here](https://saashanair.com/blog/blog-posts/deep-deterministic-policy-gradient-ddpg-how-does-the-algorithm-work), [here](https://keras.io/examples/rl/ddpg_pendulum/), and [here](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b).

The agent consists of a couple components worth mentioning:
 - Experience Replay (not prioritized)
 - Local and Target network (and corresponding soft update)
 - OUNoise



#### Experience Replay Buffer

A standard `ReplayBuffer` class described in [agent.py](./agent.py) was used.


#### Local and Target network (and corresponding soft update)

Select best action with one set of params and evaluate that action with a
different set of parameters, this is best described here: [Deep Reinforcement
Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).

Update procedure in pytorch:
```python
# tau: interpolation parameter

# iterate params and update target params with tau 
# regulated combination of local and target
for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    target_param.data.copy_(
        tau * local_param.data + (1.0 - tau) * target_param.data
    )
```

Initially, the networks are set to be the same:
```python
for target_param, source_param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(source_param.data)
```


#### OUNoise


```python
class OUNoise:
    """Ornstein-Uhlenbeck process.
    
    e.g. https://arxiv.org/pdf/1509.02971.pdf
    """

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size  # (n_agents, state_size)
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(
            size=self.size
        )
        self.state = x + dx
        return self.state
```

```python
# epsilon used to determine whether to add noise
if add_noise and (np.random.random() < self.eps):
    action += self.noise.sample()
```

governed by:
```python
self.eps = max(self.eps_end, self.eps_decay * self.eps)
```

in this case:
```python
init=1.0,
end=0.01,
decay=0.995,
```

### Config

```yaml
trainer:
  _target_: trainer.Trainer
  # episode
  n_episodes: 1000
  max_t: 3000
  
  batch_size: 64
  gamma: 0.99  # discount factor
  tau: 0.001  # soft update of target parameters

  # logistical
  n_agents: 20
  run_headless: True
  print_setup: True
  print_every: 10
  target_score: 31
  window_size: 100
  scores_path: scores.pkl

  # rpb
  buffer_size: 1000000  # replay buffer size

  agent_cfg:
    actor:
      lr: 0.0001
      hidden_units: [512, 256, 128, 64]
    critic:
      lr: 0.0003
      weight_decay: 0
      fc_1: 384
      fc_2: 256
      fc_3: 128
    eps:
      init: 1.0
      end: 0.01
      decay: 0.995
    learn_iterations: 15
    update_every: 10
    seed: 2
  
  oun_cfg:
    mu: 0.0
    theta: 0.15
    sigma: 0.2
```

## Future Work

Below are a few ideas related to improving performance

### Improving the Agent

Other Algorithms
> There are other algorithms that could be attempted
> - [D4PG](https://arxiv.org/abs/1804.08617v1)
> - [PPO](https://arxiv.org/abs/1707.06347)
> - [A3C](https://arxiv.org/abs/1602.01783)

"Prioritized" Replay Buffer?
> I'd like to think about which episodes are being saved to the replay buffer,
> rather than any/all episodes. Some episodes are likely more useful than
> others. [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) looks
> like a promising place to start

More advanced models / Stability?
> I feel like all my standard DL intuitions aren't super useful in RL. I don't
> understand this yet. For example, using "better" models doesn't seem to lead
> to "better" results. But I'll need more experience here.

Hyperparameter optimization
> This was my first time using hydra. I liked using this (though I could improve
> the organization a bit...) and I'd like to connect this to optuna. I
> personally think hyperparameter optimization is often used prematurely, but
> I'd still like to explore this option in the future.
