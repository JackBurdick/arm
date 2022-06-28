import pickle
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent


class Trainer:
    def __init__(
        self,
        n_episodes=1000,
        max_t=1000,
        buffer_size=int(1e6),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        agent_cfg=None,
        oun_cfg=None,
        target_score=30,
        n_agents=20,
        run_headless=True,
        print_setup=True,
        print_every=10,
        window_size=100,
        scores_path="scores.pkl",
    ):
        """Create an environment and train an agent

        len(env_info.agents) == 20
        action_size = brain.vector_action_space_size == 4
        state = env_info.vector_observations[0] -->
            [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
            -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
            0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
            0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
            1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
            0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
            0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
            5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
            -1.68164849e-01]
        len(state) == 33

        Parameters
        ----------
        n_episodes : int, optional
            number of episodes to train the agent, by default 1000
        max_t : int, optional
            maximum number of timesteps in an episode, by default 1000
        buffer_size : int, optional
            size of the replay buffer, by default int(1e6)
        batch_size : int, optional
            size of the batch for training, by default 64
        gamma : float, optional
            discount factor, by default 0.99
        tau : float, optional
            soft update of target parameters, by default 1e-3
        agent_cfg : dict, optional
            configuration for the agent, by default None
        oun_cfg : dict, optional
            configuration for the oun, by default None
        target_score : int, optional
            target score for the environment to be solved, by default 30
        n_agents : int, optional
            number of agents in the environment, by default 20
        run_headless : bool, optional
            run the environment in headless mode, by default True
        print_setup : bool, optional
            print the setup, by default True
        print_every : int, optional
            print the score every n episodes, by default 10
        window_size : int, optional
            size of the window for the moving average, by default 100
        """

        self.n_episodes = n_episodes
        self.max_t = max_t

        self.target_score = target_score  # 31
        self.window_size = window_size
        self.scores = []
        self.scores_window = deque(maxlen=self.window_size)
        self.print_every = print_every

        self.n_agents = n_agents

        self.save_path_fmt_score = "./params/best_params_{}_"
        self.save_scores_path = scores_path

        # environment
        if run_headless:
            self.env = UnityEnvironment(
                file_name="/home/jackburdick/dev/arm/ignore/environments/Reacher_Linux_NoVis/Reacher.x86_64"
            )
        else:
            self.env = UnityEnvironment(
                file_name="/home/jackburdick/dev/arm/ignore/environments/Reacher_Linux/Reacher.x86_64"
            )

        # get the default brain
        brain_name = self.env.brain_names[0]
        brain = self.env.brains[brain_name]

        # reset the environment
        env_info = self.env.reset(train_mode=True)[brain_name]
        self.action_size = brain.vector_action_space_size
        _ex_states = env_info.vector_observations
        self.state_size = _ex_states.shape[1]  # from first agent
        if print_setup:
            print(f"num agents: {self.n_agents}")
            print(f"state_size: {self.state_size}")
            print(f"action_size: {self.action_size}")
            print(f"example states:\n {_ex_states[0]}")

        # create agent
        self.agent = Agent(
            state_size=self.state_size,
            action_size=self.action_size,
            n_agents=self.n_agents,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tau=tau,
            agent_cfg=agent_cfg,
            oun_cfg=oun_cfg,
        )

    def _check_done_save_params(self, e):
        """Check if the environment is solved and save the parameters"""
        if np.mean(self.scores_window) >= self.target_score:
            print(
                f"\nEnv solved in {e:d} episodes!\tAvg Score: {np.mean(self.scores_window):.2f}"
            )
            save_path = self.save_path_fmt_score.format(
                int(np.mean(self.scores_window))
            )
            torch.save(
                self.agent.actor_local.state_dict(), f"{save_path}checkpoint_actor.pth"
            )
            torch.save(
                self.agent.critic_local.state_dict(),
                f"{save_path}checkpoint_critic.pth",
            )
            print(f"params saved: {save_path}....")
            # perserve score log
            with open(self.save_scores_path, "wb") as f:
                pickle.dump(self.scores, f)
            return True
        return False

    def _terminal_monitor(self, e):
        """Print the score to the terminal"""
        print(
            f"\rE: {e},\tAvg Score: {np.mean(self.scores_window) :.3f},\t\tLast Score: {np.mean(self.scores_window[-1]) : .3f}",
            end="",
        )
        if e % self.print_every == 0:
            print("\rE {}\tAvg Score: {:.2f}".format(e, np.mean(self.scores_window)))

    def _unpack_env_info(self, env_info):
        """Unpack the environment info"""
        cur_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return cur_states, rewards, dones

    def _update_score(self, scores):
        """Update the scores"""
        self.scores_window.append(scores)
        self.scores.append(np.mean(scores))

    def cleanup(self):
        """Close the environment"""
        self.env.close()
        print("env closed")

    def train(self):
        """train the agent and return the scores"""
        brain_name = self.env.brain_names[0]
        # run episode
        for e in range(1, self.n_episodes + 1):
            scores = np.zeros(self.n_agents)

            # reset
            self.agent.reset()
            env_info = self.env.reset(train_mode=True)[brain_name]

            # set initial states
            states, _, _ = self._unpack_env_info(env_info)
            for _ in range(self.max_t):

                # use states to determine action
                actions = self.agent.act(states)

                # send the action to the environment
                env_info = self.env.step([actions])[brain_name]

                # unpack reward and next states
                next_states, rewards, dones = self._unpack_env_info(env_info)

                # record step information to agent, possibly learn
                self.agent.step(states, actions, rewards, next_states, dones)

                # update state
                states = next_states

                scores += np.array(rewards)
                if any(dones):
                    break

            # update score, display in terminal
            self._update_score(scores)
            self._terminal_monitor(e)

            # Maybe save params
            if self._check_done_save_params(e):
                break

        return self.scores


if __name__ == "__main__":
    t = Trainer()
    scores = t.train()
    print(f"done: {scores}")
