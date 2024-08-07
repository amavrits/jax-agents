import os
import time
import jax
import optax
import gymnax
import numpy as np
from jaxagents import vpg
from cartpole_nn_gallery import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    """Define configuration for agent training"""
    config = vpg.AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=1_000,
        batch_size=16,
        optimizer=optax.adam,
        eval_rng=jax.random.PRNGKey(18)
    )

    """Set up agent"""
    agent = vpg.ReinforceAgent(env, env_params, config)
    print(agent.__str__())

    """Define optimizer parameters and training hyperparameters"""
    optimizer_params = vpg.OptimizerParams(learning_rate=1e-2, eps=1e-3, grad_clip=1)
    hyperparams = vpg.HyperParameters(
        gamma=0.99,
        gae_lambda=1.0,
        ent_coeff=0.01,
        # Irrelevant for this agent but helps with using the same optimizer params for critic and network. If you set
        # vf_coeff=1 and adjust the critic optimizer params to be double, leads to the same results.
        vf_coeff=0.5,
        actor_optimizer_params=optimizer_params,
        critic_optimizer_params=optimizer_params
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
    fig.savefig(os.path.join(os.getcwd(), r'figures\PG_REINFORCE training.png'))
