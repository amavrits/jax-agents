import os
import time
import distrax
import jax
import optax
import gymnax
import numpy as np
from jaxagents.ppo import PPOAgent, TrainState, STATE_TYPE, AgentConfig, OptimizerParams, HyperParameters
from jaxtyping import Float, Int, Array, PRNGKeyArray
from cartpole_nn_gallery import *
from functools import partial
import sys
import matplotlib.pyplot as plt


class CartpolePPO(PPOAgent):

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, training: TrainState, state: STATE_TYPE)-> Float[Array, "1"]:
        logits = training.apply_fn(training.params, state).squeeze()
        pis = distrax.Categorical(logits=logits)
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0, 4,))
    def _log_prob(self, training: TrainState, params: dict, state: STATE_TYPE, actions: Int[Array, "1"])\
            -> Float[Array, "1"]:
        logits = training.apply_fn(params, state).squeeze()
        log_probs = distrax.Categorical(logits=logits).log_prob(actions)
        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, training: TrainState, state: STATE_TYPE) -> Float[Array, "1"]:
        logits = training.apply_fn(jax.lax.stop_gradient(training.params), state).squeeze()
        return jnp.argmax(logits)

    @partial(jax.jit, static_argnums=(0,))
    def _sample_action(self, rng: PRNGKeyArray, training: TrainState, state: STATE_TYPE) -> Float[Array, "1"]:
        logits = training.apply_fn(jax.lax.stop_gradient(training.params), state).squeeze()
        actions = distrax.Categorical(logits=logits).sample(seed=rng)
        return actions



def plot_loss(training_metrics, eval_frequency, env_params, path):
    actor_loss = training_metrics["actor_loss"].squeeze()
    critic_loss = training_metrics["critic_loss"].squeeze()
    steps = jnp.arange(actor_loss.size) * eval_frequency
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(steps, actor_loss, c="b")
    axs[0].set_ylabel("Actor loss [-]", fontsize=12)
    axs[1].plot(steps, critic_loss, c="b")
    axs[1].set_xlabel("Training steps", fontsize=12)
    axs[1].set_ylabel("Critic loss [-]", fontsize=12)
    for ax in axs.flatten():
        ax.grid()
    plt.close()
    fig.savefig(path)


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    if sys.platform == "win32":
        checkpoint_dir = 'C:\\Users\\mavritsa\\Repositories\\jax-agents\\benchmarks\\rl\\cartpole v1\\checkpoints\\ppo'
    else:
        checkpoint_dir = '/mnt/c/Users/mavritsa/Repositories/jax-agents/benchmarks/rl/cartpole v1/checkpoints/ppo'

    """Define configuration for agent training"""
    config = AgentConfig(
        actor_network=PGActorNN,
        critic_network=PGCriticNN,
        rollout_length=50,
        n_steps=1_000,
        batch_size=16,
        minibatch_size=4,
        actor_epochs=10,
        critic_epochs=10,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
        checkpoint_dir=checkpoint_dir,
        n_evals=100,
        restore_agent=False
    )

    """Set up agent"""
    agent = CartpolePPO(env, env_params, config)
    print(agent.__str__())

    """Define optimizer parameters and training hyperparameters"""
    hyperparams = HyperParameters(
        gamma=0.99,
        eps_clip=0.2,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=0.0,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=3e-4, eps=1e-3, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=1e-3, eps=1e-3, grad_clip=1)
    )
    agent.log_hyperparams(hyperparams)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Train agent"""
    t0 = time.time()
    runner, training_metrics = jax.block_until_ready(agent.train(rng_train, hyperparams))
    print(f"time: {time.time() - t0:.2f} s")

    """ Post-process results"""
    training_returns = agent.summarize(training_metrics["returns"])
    agent.collect_training(runner, training_metrics)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=16)
    eval_returns = agent.summarize(eval_metrics["returns"])
    print(eval_returns.episode_metric.min(), eval_returns.episode_metric.max())

    plot_loss(training_metrics, config.eval_frequency, env_params, r'figures/PPO losses.png')

    fig = plt.figure()
    plt.fill_between(agent.eval_steps_in_training, training_returns.min, training_returns.max, color='b', alpha=0.4)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Training reward [-]", fontsize=14)
    plt.close()
    fig.savefig(os.path.join(os.getcwd(), r'figures/PPO Clip training.png'))

