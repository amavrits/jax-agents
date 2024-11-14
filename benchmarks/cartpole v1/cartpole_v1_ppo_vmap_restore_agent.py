import os
import time
import jax
import optax
import gymnax
import numpy as np
from jaxagents import ppo
from cartpole_nn_gallery import *
from functools import partial
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    if sys.platform == "win32":
        checkpoint_dir = 'C:\\Users\\Repositories\\jax-agents\\benchmarks\\cartpole v1\\checkpoints\\ppo_vmap'
    else:
        checkpoint_dir = '/mnt/c/Users/Repositories/jax-agents/benchmarks/cartpole v1/checkpoints/ppo_vmap'

    """Define configuration for agent training"""
    config = ppo.AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=200,
        batch_size=16,
        actor_epochs=10,
        critic_epochs=10,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
        checkpoint_dir=checkpoint_dir,
        restore_agent=False
    )

    """Set up agent"""
    agent = ppo.PPOAgent(env, env_params, config)
    print(agent.__str__())

    """Define optimizer parameters and training hyperparameters"""
    hyperparams = ppo.HyperParameters(
        gamma=jnp.array([0.99, 0.99, 0.99, 0.99]),
        eps_clip=jnp.array([0.05, 0.1, 0.15, 0.2]),
        kl_threshold=jnp.array([1e-8, 1e-7, 1e-6, 1e-5]),
        gae_lambda=jnp.array([0.97, 0.97, 0.97, 0.97]),
        ent_coeff=jnp.zeros(4),
        vf_coeff=jnp.ones(4),
        actor_optimizer_params=ppo.OptimizerParams(
            learning_rate=jnp.array([5e-5, 1e-4, 2e-4, 3e-4]),
            eps=jnp.ones(4)*1e-3,
            grad_clip=jnp.ones(4)
        ),
        critic_optimizer_params=ppo.OptimizerParams(
            learning_rate=jnp.array([1e-4, 2e-4, 5e-4, 1e-3]),
            eps=jnp.ones(4)*1e-3,
            grad_clip=jnp.ones(4)
        )
    )
    agent.log_hyperparams(hyperparams)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Train agent"""
    t0 = time.time()
    vmap_train = jax.vmap(agent.train, in_axes=(None, 0))
    runner, training_metrics = jax.block_until_ready(vmap_train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """Continue training agent"""
    config_restore = ppo.AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=800,
        batch_size=16,
        actor_epochs=10,
        critic_epochs=10,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
        checkpoint_dir=checkpoint_dir,
        restore_agent=True
    )

    agent = ppo.PPOAgent(env, env_params, config_restore)
    agent.restore()

    t0 = time.time()
    # Use the same training rng, it is trivial when training with a restored agent.
    vmap_train = jax.vmap(agent.train, in_axes=(None, 0))
    runner, training_metrics = jax.block_until_ready(vmap_train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """ Post-process results"""
    training_returns = jax.vmap(agent.summarize)(training_metrics["episode_returns"])

    fig = plt.figure()
    n_parallel = training_returns.max.shape[0]
    colors = cm.rainbow(np.linspace(0, 1, n_parallel))
    for i_parallel in range(n_parallel):
        plt.fill_between(np.arange(1, 1_000+2, config.eval_frequency), training_returns.min[i_parallel],
                         training_returns.max[i_parallel], color=colors[i_parallel], alpha=0.4, label=str(i_parallel+1))
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Training reward [-]", fontsize=14)
    plt.legend(title="Hyperparameter set:", fontsize="small")
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), r'figures\PPO Clip training vmap restored.png'))
