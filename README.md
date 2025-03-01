# Jaxagents

Jaxagents is a Python implementation of Reinforcement Learning agents built upon JAX. The PyPI page of the project can be found [here](https://pypi.org/project/jaxagents/).

## Installation
You can install the latest version of jaxagents from PyPI via:

```sh
pip install jaxagents
```

## Content

So far, the project includes the following agents for Reinforcement Learning:
* Q-learning:
  * Deep Q Networks (DQN)
  * Double Deep Q Networks (DDQN) 
  * Categorical Deep Q Networks (often known as C51)
  * Quantile Regression Deep Q Networks (QRDQN) 
* Policy gradient:
  * REINFORCE
  * PPO with clipping and GAE

So far, the project includes the following agents for Multi-Agent Reinforcement Learning:
* Policy gradient:
  * IPPO

## Background

Research and development in Reinforcement Learning can be computationally cumbersome. Utilizing JAX's high computational performance, Jaxagents provides a framework for applying and developing Reinforcement Learning agents that offers benefits in:
* computational speed
* easy control of random number generation
* hyperparameter optimization (via parallelized calculations)
