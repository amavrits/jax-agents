import chex
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
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


class PGActorNNDiscrete(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        action_dim = 4

        activation = nn.tanh

        actor_mean = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        return pi


class PGCriticNN(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        activation = nn.tanh

        critic = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class PGActorNNContinuous(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        activation = nn.tanh

        actor = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation(actor)
        actor = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation(actor)
        actor = nn.Dense(2, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)

        # pi = distrax.Normal(loc=jnp.take(actor, 0), scale=jnp.exp(jnp.take(actor, 1)))
        pi = distrax.Beta(alpha=1+jnp.exp(jnp.take(actor, 0)), beta=1+jnp.exp(jnp.take(actor, 1)))
        # pi = distrax.Gamma(concentration=jnp.exp(jnp.take(actor, 0)), rate=jnp.exp(jnp.take(actor, 1)))

        return pi


class PGActorNNDiscreteMA(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        n_actors = 2
        action_dim = 4

        activation = nn.tanh

        actor_mean = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(action_dim*n_actors, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        # pi = distrax.Categorical(logits=actor_mean)

        return actor_mean.reshape(n_actors, action_dim)


class PGActorNNContinuousMA(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        n_actors = 2
        activation = nn.tanh

        actor = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation(actor)
        actor = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation(actor)
        actor = nn.Dense(n_actors*2, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)
        # actor = nn.relu(actor)

        return actor.reshape(n_actors, 2)


class PGCriticNNMA(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        n_critics = 2

        activation = nn.tanh

        critic = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(n_critics, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return critic.squeeze()

