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

    """Define optimizer parameters and training hyperparameters"""
    hyperparams = ppo.HyperParameters(
        gamma=0.99,
        eps_clip=0.2,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=0.0,
        vf_coeff=1.0,
        actor_optimizer_params=ppo.OptimizerParams(learning_rate=3e-4, eps=1e-3, grad_clip=1),
        critic_optimizer_params=ppo.OptimizerParams(learning_rate=1e-3, eps=1e-3, grad_clip=1)
    )

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=16)
    eval_rewards = agent.summarize(eval_metrics)
    print(eval_rewards.episode_metric.min(), eval_rewards.episode_metric.max())

