import os
import time
import jax
import optax
import gymnax
import numpy as np
from jaxagents import vpg as vpg
from cartpole_nn_gallery import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    """Set up transition template, given the state representation in the cartpole environment."""
    transition_temp = vpg.Transition(
        state=jnp.zeros((1, 4), dtype=jnp.float32),
        action=jnp.zeros(1, dtype=jnp.int32),
        log_prob=jnp.zeros(1, dtype=jnp.float32),
        value=jnp.zeros(1, dtype=jnp.float32),
        reward=jnp.zeros(1, dtype=jnp.float32),
        next_state=jnp.zeros((1, 4), dtype=jnp.float32),
        terminated=jnp.zeros(1, dtype=jnp.bool_),
        info={
            "discount": jnp.array((), dtype=jnp.float32),
            "returned_episode": jnp.array((), dtype=jnp.bool_),
            "returned_episode_lengths": jnp.array((), dtype=jnp.int32),
            "returned_episode_returns": jnp.array((), dtype=jnp.float32),
        }
    )

    """Define configuration for agent training"""
    config = vpg.AgentConfig(
        actor_network=VanillaPG_Actor_NN_model,
        critic_network=VanillaPG_Critic_NN_model,
        transition_template=transition_temp,
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
        gae_lambda=1,
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
    training_rewards = np.asarray(training_metrics["episode_rewards"])
    agent.collect_training(runner, training_metrics)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=500_000)
    eval_rewards = agent.summarize(eval_metrics["done"].flatten(), eval_metrics["reward"].flatten())
    print(eval_rewards.episode_metric.min(), eval_rewards.episode_metric.max())

    fig = plt.figure()
    plt.fill_between(np.arange(1, agent.config.n_steps+1), training_rewards.min(axis=1),
                     training_rewards.max(axis=1), color='b', alpha=0.4)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Training reward [-]", fontsize=14)
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), r'figs\PG_REINFORCE training.png'))
