import time
import jax
from jax import jit
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import gymnax
import optax
from jax_tqdm import scan_tqdm
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import os
import sys
sys.path.append(os.getcwd()+"/blackjax/agents")
sys.path.append(os.getcwd()+"/blackjax/utils")

from DQN_HyperparamTuning import make_train,Transition
from VizOutput import *


def eval_agent(rng, nn_params, q_network, n_steps=500_000):

    rng, _rng = jax.random.split(rng)
    _rng, reset_rng = jax.random.split(_rng)
    state, env_state = env.reset(reset_rng, env_params)

    @jit
    @scan_tqdm(n_steps)
    def _run_eval_step(runner, unused):

        env_state, state, rng = runner

        q_state = q_network.apply(nn_params, state)
        action = jnp.argmax(q_state, 1)

        rng, _rng = jax.random.split(rng)
        _rng, step_rng = jax.random.split(_rng)
        next_state, next_env_state, reward, terminated, info = env.step(step_rng, env_state, action.squeeze(),
                                                                        env_params)
        runner = (next_env_state, next_state, rng)
        metrics = {
            "done": terminated,
            "reward": reward
        }

        return runner, metrics

    rng, _rng = jax.random.split(rng)
    runner = (env_state, state, _rng)
    runner, metrics = lax.scan(_run_eval_step, runner, jnp.arange(n_steps), n_steps)

    return metrics


if __name__ == '__main__':

    base_transition = Transition(
        state=jnp.zeros((1, 4), dtype=jnp.float32),
        action=jnp.zeros(1, dtype=jnp.int32),
        reward=jnp.zeros(1, dtype=jnp.float32),
        next_state=jnp.zeros((1, 4), dtype=jnp.float32),
        terminated=jnp.zeros(1, dtype=jnp.bool_),
        info={
            " discount": jnp.array((), dtype=jnp.float32),
            "returned_episode": jnp.array((), dtype=jnp.bool_),
            "returned_episode_lengths": jnp.array((), dtype=jnp.int32),
            "returned_episode_returns": jnp.array((), dtype=jnp.float32),
        }
    )

    class NN_model(nn.Module):
        action_dim: Sequence[int]

        @nn.compact
        def __call__(self, x):

            q = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            q = nn.relu(q)
            q = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q)
            q = nn.relu(q)
            q = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q)
            q = nn.relu(q)
            q = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(q)

            return q.reshape(-1, self.action_dim)


    env, env_params = gymnax.make("CartPole-v1")

    config = {
        "Q-MODEL": NN_model,
        "TOTAL_STEPS": 2_000_000,
        "BATCH_SIZE": 32,
        "BUFFER_SIZE": 100_000,
        "EPS_FN": lambda t: 0.01 + (1 - 0.01) * jnp.exp(-t / 50_000),
        "GAMMA": 0.99,
        "ENV": env,
        "ENV_PARAMS": env_params,
        "BASE_TRANSITION": base_transition,
        "AGENT_EVAL_FN": lambda runner: 0,
        "MAX_GRAD_NORM": 100,
        "ANNEAL_LR": False,
        "LOSS_FN": optax.l2_loss,
        # "LOSS_FN": optax.huber_loss,
        # "OPTIMIZER": optax.adam,
        "OPTIMIZER": optax.adamw,
        # "OPTIMIZER": optax.rmsprop,
    }

    # lr_grid = jnp.array([1e-2, 1e-3, 1e-4, 1e-5])
    # traget_period_grid = jnp.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000])
    lr_grid = jnp.array([1e-3, 1e-4, 1e-5])
    traget_period_grid = jnp.array([10, 50, 100, 200])
    # buffer_grid = jnp.array([1e3, 1e4, 1e5, 1e6])
    # batch_grid = jnp.array([32, 64, 128, 256, 512])

    hyperpars = jnp.meshgrid(lr_grid, traget_period_grid)
    hyperpars = jnp.c_[[item.flatten() for item in hyperpars]].T

    rng = jax.random.PRNGKey(42)
    t0 = time.time()


    train_vvjit = jax.jit(jax.vmap(make_train(config), in_axes=(0, None)))
    outs = jax.block_until_ready(train_vvjit(hyperpars, rng))
    print(f"time: {time.time() - t0:.2f} s")
    df = collect_results_hyperparameters(outs, hyperpars)
    df.to_csv(os.getcwd()+"/results/cartpole v1/Hyperparameter_Tuning_Cartpole.csv", index=False)
    plot_agents_hyperpars(df, os.getcwd()+"/results/cartpole v1/Hyperparameter_Tuning_Cartpole.pdf", 100)


    # rngs = jax.random.split(rng, 8)
    # train_vvjit = jax.jit(jax.vmap(jax.vmap(make_train(config), in_axes=(None, 0)), in_axes=(0, None)))
    # outs = jax.block_until_ready(train_vvjit(hyperpars, rngs))
    # print(f"time: {time.time() - t0:.2f} s")
    # df = collect_results_hyperparameters_seeds(outs, hyperpars)
    # df.to_csv(os.getcwd()+"/results/cartpole v1/Hyperparameter_Tuning_Seeds_Cartpole.csv", index=False)
    # plot_agents_hyperpars_seeds(df, os.getcwd()+"/results/cartpole v1/Hyperparameter_Tuning_Seeds_Cartpole.pdf", 100)

