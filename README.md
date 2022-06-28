[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Arm

### Introduction

For this project, you will work with the [Reacher](https://github.com/deepanshut041/Reinforcement-Learning/tree/master/mlagents/04_reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Environment Information

Information adapted from [here](https://github.com/deepanshut041/Reinforcement-Learning/tree/master/mlagents/04_reacher)

- Set-up: Double-jointed arm which can move to target locations.
- Goal: The agents must move its hand to the goal location, and keep it there.
- Agents: The environment contains 10 agent with same Behavior Parameters.
- Agent Reward Function (independent):
  - +0.1 Each step agent's hand is in goal location.
- Behavior Parameters:
  - Vector Observation space: 26 variables corresponding to position, rotation,
    velocity, and angular velocities of the two arm Rigidbodies.
  - Vector Action space: (Continuous) Size of 4, corresponding to torque
    applicable to two joints.
  - Visual Observations: None.
- Float Properties: Five
  - goal_size: radius of the goal zone
    - Default: 5
    - Recommended Minimum: 1
    - Recommended Maximum: 10
  - goal_speed: speed of the goal zone around the arm (in radians)
    - Default: 1
    - Recommended Minimum: 0.2
    - Recommended Maximum: 4
  - gravity
    - Default: 9.81
    - Recommended Minimum: 4
    - Recommended Maximum: 20
  - deviation: Magnitude of sinusoidal (cosine) deviation of the goal along the vertical dimension
    - Default: 0
    - Recommended Minimum: 0
    - Recommended Maximum: 5
  - deviation_freq: Frequency of the cosine deviation of the goal along the vertical dimension
    - Default: 0
    - Recommended Minimum: 0
    - Recommended Maximum: 3

### Goal

For this project, the Unity environment contains 20 identical agents, each with its own copy of the environment.  

Agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### Getting Started (Dependencies)

- Unity Ml Agents
- PyTorch
- numpy
- matplotlib
- hydra (`pip install hydra-core --upgrade`)


### Instructions

To set up the environment please see [unity_instructions](./unity_instructions.md)

The main entry is `arm.ipynb`

### Contents

- [arm.ipynb](./arm.ipynb)
    - main development environment
- [config.yaml](./conf/config.yaml)
    - main configuration file
- [trainer.py](./trainer.py)
    - Convenience wrapper to execute training of an agent in the environment
- [agent.py](./agent.py)
    - The convenience wrapper that uses the model (specified below) to learn and interact with the environment
- [model.py](./model.py)
    - The model used to predict actions from the environment
- [/params/best_params_31_checkpoint_actor.pth](./params/best_params_31_checkpoint_actor.pth)
    - model weights saved from a trained actor
- [/params/best_params_31_checkpoint_critic.pth](./params/best_params_31_checkpoint_critic.pth)
    - model weights saved from a trained critic
- [scores.pkl](./scores.pkl)
    - log of scores over time during training run