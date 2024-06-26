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
        rollout_length=40,
        n_steps=100_000,
        update_epochs=1,
        batch_size=8,
        store_agent=False,
        act_randomly=lambda random_key, state, n_actions: jax.random.choice(random_key, jnp.arange(n_actions)),
        get_performance=lambda i_step, step_runner: 0,
        optimizer=optax.rmsprop,
        loss_fn=optax.l2_loss
    )


    """Set up agent"""
    agent = vpg.VPGAgent(env, env_params, config)
    print(agent.__str__())


    """Define optimizer parameters and training hyperparameters"""
    optimizer_params = vpg.OptimizerParams(learning_rate=5e-3, eps=1e-3, grad_clip=1)
    hyperparams = vpg.HyperParameters(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.05,
        vf_coeff=0.5,
        ent_coeff=0.01,
        actor_optimizer_params=optimizer_params,
        critic_optimizer_params=optimizer_params
    )


    """Draw random key"""
    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Train agent"""
    t0 = time.time()
    # runner, training_metrics = jax.block_until_ready(agent.train(rng_train, hyperparams))
    with jax.disable_jit(True): runner, training_metrics = jax.block_until_ready(agent.train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """ Post-process results"""
    agent.collect_training(runner)
    training_rewards = agent.summarize(training_metrics["done"].flatten(), training_metrics["reward"].flatten())

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=500_000)
    eval_rewards = agent.summarize(eval_metrics["done"].flatten(), eval_metrics["reward"].flatten())

    """ Plot results"""
    running_window = 100
    running_training_rewards = (np.cumsum(training_rewards.episode_metric)[running_window:] -
                                np.cumsum(training_rewards.episode_metric)[:-running_window]) / running_window
    running_eval_rewards = (np.cumsum(eval_rewards.episode_metric)[running_window:] -
                                np.cumsum(eval_rewards.episode_metric)[:-running_window]) / running_window

    fig = plt.figure()
    plt.plot(training_rewards.episode_metric, c='b', alpha=0.4)
    plt.plot(np.arange(running_window, training_rewards.episode_metric.size), running_training_rewards, c='b')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward [-]", fontsize=14)
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), r'figs\vpg training.png'))
