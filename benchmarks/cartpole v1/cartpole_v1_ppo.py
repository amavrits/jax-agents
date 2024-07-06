import os
import time
import jax
import optax
import gymnax
import numpy as np
from jaxagents import ppo
from cartpole_nn_gallery import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

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
        eval_rng=jax.random.PRNGKey(18)
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
        actor_optimizer_params=ppo.OptimizerParams(learning_rate=3e-4),
        critic_optimizer_params=ppo.OptimizerParams(learning_rate=1e-3)
    )

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Train agent"""
    t0 = time.time()
    runner, training_metrics = jax.block_until_ready(agent.train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """ Post-process results"""
    training_rewards = agent.summarize(training_metrics["episode_rewards"])
    agent.collect_training(runner, training_metrics)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=16)
    eval_rewards = agent.summarize(eval_metrics)
    print(eval_rewards.episode_metric.min(), eval_rewards.episode_metric.max())

    fig = plt.figure()
    plt.fill_between(np.arange(1, agent.config.n_steps+1), training_rewards.episode_metric.min(axis=1),
                     training_rewards.episode_metric.max(axis=1), color='b', alpha=0.4)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Training reward [-]", fontsize=14)
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), r'figures\PPO Clip training.png'))
