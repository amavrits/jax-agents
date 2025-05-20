import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, lecun_normal, variance_scaling
from typing import Sequence
import jax.numpy as jnp
import distrax


class DQN_NN(nn.Module):
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


class Categorical_NN(nn.Module):
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


class QRDDQN_NN(nn.Module):
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


# class PGActorNN(nn.Module):
#     config: dict
#
#     @nn.compact
#     def __call__(self, x):
#
#         action_dim = 2
#         activation = nn.tanh
#
#         logits = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         logits = activation(logits)
#         logits = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(logits)
#         logits = activation(logits)
#         logits = nn.Dense(action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(logits)
#
#         return logits
#
#
# class PGCriticNN(nn.Module):
#     config: dict
#
#     @nn.compact
#     def __call__(self, x):
#
#         action_dim = 2
#         activation = nn.tanh
#
#         critic = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         critic = activation(critic)
#         critic = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critic)
#         critic = activation(critic)
#         critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
#
#         return jnp.squeeze(critic, axis=-1)



class PGActorNN(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):
        action_dim = 2
        activation = nn.tanh

        init1 = variance_scaling(jnp.sqrt(2), 'fan_avg', 'truncated_normal')
        init2 = variance_scaling(jnp.sqrt(2), 'fan_avg', 'truncated_normal')
        init3 = variance_scaling(0.01, 'fan_avg', 'truncated_normal')

        logits = nn.Dense(128, kernel_init=init1, bias_init=constant(0.0))(x)
        logits = activation(logits)
        logits = nn.Dense(64, kernel_init=init2, bias_init=constant(0.0))(logits)
        logits = activation(logits)
        logits = nn.Dense(action_dim, kernel_init=init3, bias_init=constant(0.0))(logits)

        return logits


class PGCriticNN(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):
        activation = nn.tanh

        init1 = variance_scaling(jnp.sqrt(2), 'fan_avg', 'truncated_normal')
        init2 = variance_scaling(jnp.sqrt(2), 'fan_avg', 'truncated_normal')
        init3 = variance_scaling(1.0, 'fan_avg', 'truncated_normal')

        critic = nn.Dense(128, kernel_init=init1, bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=init2, bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=init3, bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)

