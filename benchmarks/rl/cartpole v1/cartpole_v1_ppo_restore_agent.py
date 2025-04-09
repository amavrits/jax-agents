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
        checkpoint_dir = 'C:\\Users\\Repositories\\jax-agents\\benchmarks\\rl\\cartpole v1\\checkpoints\\ppo'
    else:
        checkpoint_dir = '/mnt/c/Users/Repositories/jax-agents/benchmarks/rl/cartpole v1/checkpoints/ppo'

    """Define configuration for agent training"""
    config = ppo.AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=30,
        batch_size=16,
        minibatch_size=4,
        actor_epochs=10,
        critic_epochs=10,
        optimizer=optax.adam,
        eval_frequency=1,
        eval_rng=jax.random.PRNGKey(18),
        checkpoint_dir=checkpoint_dir,
        n_evals=100,
        restore_agent=False
    )

    """Set up agent"""
    agent = ppo.PPOAgent(env, env_params, config)
    print(agent.__str__())

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

    """Train agent"""
    t0 = time.time()
    runner, training_metrics = jax.block_until_ready(agent.train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """Restore agent"""
    config_restore = ppo.AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=70,
        batch_size=16,
        minibatch_size=4,
        actor_epochs=10,
        critic_epochs=10,
        optimizer=optax.adam,
        eval_frequency=1,
        eval_rng=jax.random.PRNGKey(18),
        checkpoint_dir=checkpoint_dir,
        n_evals=100,
        restore_agent=True
    )

    agent = ppo.PPOAgent(env, env_params, config_restore)
    agent.restore(mode='last')

    """Continue training agent"""
    t0 = time.time()
    # Use the same training rng, it is trivial when training with a restored agent.
    runner, training_metrics = jax.block_until_ready(agent.train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """ Post-process results"""
    training_returns = agent.summarize(training_metrics["episode_returns"])
    agent.collect_training(runner, training_metrics)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=16)
    eval_returns = agent.summarize(eval_metrics)
    print(eval_returns.episode_metric.min(), eval_returns.episode_metric.max())

    fig = plt.figure()
    idx = jnp.where(agent.eval_steps_in_training <= config.n_steps)
    plt.fill_between(agent.eval_steps_in_training[idx], training_returns.min[idx], training_returns.max[idx], color='b',
                     alpha=0.4, label='Initial training')
    idx = jnp.where(agent.eval_steps_in_training >= config.n_steps)
    plt.fill_between(agent.eval_steps_in_training[idx], training_returns.min[idx], training_returns.max[idx], color='r',
                     alpha=0.4, label='Continued training')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Training reward [-]", fontsize=14)
    plt.legend(fontsize=12)
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), r'figures/PPO Clip training restored.png'))
