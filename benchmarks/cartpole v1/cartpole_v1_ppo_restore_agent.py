import os
import time
import jax
import optax
import gymnax
import numpy as np
from jaxagents import ppo
from cartpole_nn_gallery import *
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    if sys.platform == "win32":
        checkpoint_dir = 'C:\\Users\\Repositories\\jax-agents\\benchmarks\\cartpole v1\\checkpoints\\ppo'
    else:
        checkpoint_dir = '/mnt/c/Users/Repositories/jax-agents/benchmarks/cartpole v1/checkpoints/ppo'

    """Define configuration for agent training"""
    config = ppo.AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=1_000,
        batch_size=16,
        actor_epochs=10,
        critic_epochs=10,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
        checkpoint_dir=checkpoint_dir,
        restore_agent=True
    )

    """Set up agent"""
    agent = ppo.PPOAgent(env, env_params, config)
    print(agent.__str__())

    agent.restore()
    training_metrics = agent.training_metrics

    """ Post-process results"""
    training_returns = agent.summarize(training_metrics["episode_returns"])
    agent.collect_training(agent.training_runner, training_metrics)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=16)
    eval_rewards = agent.summarize(eval_metrics)
    print(eval_rewards.episode_metric.min(), eval_rewards.episode_metric.max())

