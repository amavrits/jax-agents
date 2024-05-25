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
# sys.path.append(os.getcwd()+"/blackjax/agents")
# sys.path.append(os.getcwd()+"/blackjax/utils")
sys.path.append("C:\\Users\\Repositories\\BlackJax_RL\\blackjax\\agents")
sys.path.append("C:\\Users\\Repositories\\BlackJax_RL\\blackjax\\utils")

# from DQN2 import make_train, Transition
from DDQN import make_train, Transition
from VizOutput import *


if __name__ == '__main__':



    class NN_model(nn.Module):
        action_dim: Sequence[int]

        @nn.compact
        def __call__(self, x):

            q = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            q = nn.relu(q)
            q = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q)
            q = nn.relu(q)
            q = nn.Dense(32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(q)
            q = nn.relu(q)
            q = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(q)

            return q.reshape(-1, self.action_dim)


    # from jax.config import config
    # config.update('jax_disable_jit', True)


    env, env_params = gymnax.make("CartPole-v1")

    config = {
        "STATE_SIZE": 4,
        "INIT_NN_PARAMS": None,
        "Q-MODEL": NN_model,
        "TOTAL_STEPS": 500_000,
        "NUM_ENVS": 1,
        "BATCH_SIZE": 128,
        "BUFFER_SIZE": 10_000,
        "TAU": 0.005,
        "TARGET_UPDATE_PERIOD": 100,
        "INIT_BUFFER": None,
        "RANDOM_ACTION_FN": lambda rng, state: jax.random.choice(rng, jnp.arange(2)),
        "EPS_FN": lambda i_episode: 0.05 + (0.90 - 0.05) * jnp.exp(-i_episode / 50_000),
        "GAMMA": 0.99,
        "LR": 1e-4,
        "ENV": env,
        "ENV_PARAMS": env_params,
        "STORE_AGENT": False,
        "BASE_TRANSITION": base_transition,
        "AGENT_EVAL_FN": lambda i_step, runner: 0,
        "MAX_GRAD_NORM": 1,
        "ANNEAL_LR": False,
        "LOSS_FN": optax.l2_loss,
        # "LOSS_FN": optax.huber_loss,
        # "OPTIMIZER": optax.adam,
        # "OPTIMIZER": optax.adamw,
        "OPTIMIZER": optax.rmsprop,
        "OPT_EPS": 1e-5,
    }


    train_jit = jax.jit(make_train(config))
    rng = jax.random.PRNGKey(42)
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    print(f"time: {time.time() - t0:.2f} s")


    N = 100
    rewards = extract_rewards(out["metrics"]["done"], out["metrics"]["reward"])
    running_rewards = (jnp.cumsum(rewards)[N:] - jnp.cumsum(rewards)[:-N]) / N
    fig = plt.figure()
    plt.plot(rewards, c='b', alpha=0.4)
    plt.plot(jnp.arange(N, rewards.size), running_rewards, c='b')


    A = np.asarray(out["metrics"]["dummy"].squeeze())

