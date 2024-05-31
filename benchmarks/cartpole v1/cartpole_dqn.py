import sys
import os
import time
import jax
import optax
import gymnax
import numpy as np
from cartpole_nn_gallery import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(sys.path[2], 'jaxagents', 'agents'))
try:
    import dqn
except:
    raise


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    """Set up transition template, given the state representation in the cartpole environment"""
    transition_temp = dqn.Transition(
        state=jnp.zeros((1, 4), dtype=jnp.float32),
        action=jnp.zeros(1, dtype=jnp.int32),
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

    """Set up function for initializing the optimizer"""
    def optimizer_fn(optimizer_params):
        return optax.chain(
            optax.clip_by_global_norm(optimizer_params.grad_clip),
            optax.rmsprop(learning_rate=optimizer_params.learning_rate, eps=optimizer_params.eps)
            )

    """Define configuration for agent training"""
    config = dqn.AgentConfig(
        q_network=DQN_NN_model,
        transition_template=transition_temp,
        n_steps=500_000,
        buffer_type="FLAT",
        buffer_size=10_000,
        batch_size=128,
        target_update_method="PERIODIC",
        store_agent=False,
        act_randomly=lambda random_key, state, n_actions: jax.random.choice(random_key, jnp.arange(n_actions)),
        get_performance=lambda i_step, step_runner: 0,
        optimizer=optax.rmsprop,
        loss_fn=optax.l2_loss,
        epsilon_fn_style="DECAY",
        epsilon_params=(0.9, 0.05, 50_000)
    )


    """Set up agent"""
    agent = dqn.DDQN_Agent(env, env_params, config)
    print(agent.__str__())


    """Define optimizer parameters and training hyperparameters"""
    optimizer_params = dqn.OptimizerParams(learning_rate=5e-3, eps=1e-3, grad_clip=1)
    hyperparams = dqn.HyperParameters(0.99, 4, optimizer_params)


    """Draw random key"""
    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)


    """Train agent"""
    t0 = time.time()
    runner, training_metrics = agent.train(rng_train, hyperparams)
    print(f"time: {time.time() - t0:.2f} s")


    """ Post-process results"""
    agent.collect_training(runner)
    training_rewards = agent.summarize(training_metrics["done"], training_metrics["reward"])
    buffer_export = agent.export_buffer()


    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=500_000)
    eval_rewards = agent.summarize(eval_metrics["done"], eval_metrics["reward"])


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
    fig.savefig(sys.path[0]+r'\figs\DDQN training.png')
