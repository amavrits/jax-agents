import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import jax.numpy as jnp


class DQN_NN_model(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, x):
        n_atoms = 51

        q = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        q = nn.relu(q)
        q = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(q)

        return q.reshape(-1, self.action_dim)


class Categorical_NN_model(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, x):

        n_atoms = self.config["N_ATOMS"]

        q = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        q = nn.relu(q)
        q = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(self.action_dim * n_atoms, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(q)

        return q.reshape(-1, self.action_dim, n_atoms)


class QRDDQN_NN_model(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, x):

        n_quantiles = self.config["N_QUANTILES"]

        x = x.reshape(-1, x.shape[-1])

        q = x
        q = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(q)
        q = nn.relu(q)
        q = nn.Dense(self.action_dim * n_quantiles, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(q)
        q = q.reshape(-1, self.action_dim, n_quantiles)

        return q
