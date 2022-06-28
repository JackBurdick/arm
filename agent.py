import copy
import random
from collections import deque, namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        n_agents: int,
        buffer_size: int = int(1e6),
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-3,
        agent_cfg: Any = None,
        oun_cfg: Any = None,
    ):
        """Initialize an Agent object

        Parameters
        ----------
        state_size : int
            size of state space
        action_size : int
            size of action space
        n_agents : int
            number of agents
        buffer_size : int, optional
            size of replay buffer, by default int(1e6)
        batch_size: int, optional
            size of batch for learning, by default 64
        gamma : float, optional
            discount factor, by default 0.99
        tau : float, optional
            soft update of target parameters, by default 1e-3
        agent_cfg : Any, optional
            config for agent, by default None
        oun_cfg : Any, optional
            config for oun, by default None
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = agent_cfg.seed

        # replay buffer
        self.buffer_size = buffer_size

        self.batch_size = batch_size

        self.gamma = gamma

        self.eps = agent_cfg.eps.init
        self.eps_end = agent_cfg.eps.end
        self.eps_decay = agent_cfg.eps.decay

        # soft update
        self.tau = tau

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size,
            action_size,
            hidden_units=agent_cfg.actor.hidden_units,
            seed=self.seed,
        ).to(device)

        self.actor_target = Actor(
            state_size,
            action_size,
            hidden_units=agent_cfg.actor.hidden_units,
            seed=self.seed,
        ).to(device)

        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=agent_cfg.actor.lr
        )

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size,
            action_size,
            fc_1=agent_cfg.critic.fc_1,
            fc_2=agent_cfg.critic.fc_2,
            fc_3=agent_cfg.critic.fc_3,
            seed=self.seed,
        ).to(device)

        self.critic_target = Critic(
            state_size,
            action_size,
            fc_1=agent_cfg.critic.fc_1,
            fc_2=agent_cfg.critic.fc_2,
            fc_3=agent_cfg.critic.fc_3,
            seed=self.seed,
        ).to(device)

        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=agent_cfg.critic.lr,
            weight_decay=agent_cfg.critic.weight_decay,
        )

        # Noise process
        self.noise = OUNoise(
            (self.n_agents, self.action_size),
            self.seed,
            mu=oun_cfg.mu,
            theta=oun_cfg.theta,
            sigma=oun_cfg.sigma,
        )

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, self.buffer_size, self.batch_size, self.seed
        )

        self.t_step = 0
        self.update_every = agent_cfg.update_every
        self.learn_iterations = agent_cfg.learn_iterations

        # set local and target to be eq initially
        self.copy_params(self.actor_target, self.actor_local)
        self.copy_params(self.critic_target, self.critic_local)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experiences from each agent in replay memory, and use random sample from buffer to learn."""

        # Save experience and reward from each agent
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):

            self.memory.add(state, action, reward, next_state, done)

        # Learn
        self.t_step += 1
        if self.t_step % self.update_every == 0 and len(self.memory) > self.batch_size:
            for _ in range(self.learn_iterations):
                # TODO: parameterize
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()

        # get current action per present policy
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # include noise
        if add_noise and (np.random.random() < self.eps):
            # decrease epsilon
            self.eps = max(self.eps_end, self.eps_decay * self.eps)
            action += self.noise.sample()

        # clip action
        action = np.clip(action, -1, 1)

        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
        experiences : Tuple[torch.Tensor]
            tuple of (s, a, r, s', done) tuples
        gamma : float
            discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # clip gradients before step
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model : PyTorch model
            weights will be copied from
        target_model : PyTorch model
            weights will be copied to
        tau : float
            interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def copy_params(self, target_model, source_model):
        """Copy model parameters from source to target model."""
        for target_param, source_param in zip(
            target_model.parameters(), source_model.parameters()
        ):
            target_param.data.copy_(source_param.data)


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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        """Initialize a ReplayBuffer object

        Parameters
        ----------
        action_size : int
            size of action space
        buffer_size : int
            size of buffer
        batch_size : int
            size of batch
        seed : int
            random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
