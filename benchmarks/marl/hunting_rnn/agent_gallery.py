import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp


class PGActorDiscrete(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, h):

        n_actors = 2
        action_dim = 4
        hidden_size = self.config.hidden_size

        for t in range(x.shape[1]):
            h = nn.tanh(nn.Dense(hidden_size)(jnp.concatenate([x[:, t, :], jnp.expand_dims(h, axis=0)], axis=-1))).squeeze()
        logits = nn.Dense(action_dim*n_actors)(h)

        return logits.reshape(n_actors, action_dim), h


class PGActorContinuous(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, h):

        n_actors = 2
        # action_dim = 4
        activation = nn.tanh
        hidden_size = self.config.hidden_size
        batch_size, seq_length, n_features = x.shape

        for t in range(seq_length):
            h = activation(nn.Dense(hidden_size)(jnp.concatenate([x[:, t, :], jnp.expand_dims(h, axis=0)], axis=-1))).squeeze()
        actors = nn.Dense(n_actors)(h)

        return actors, h


class PGCritic(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, h):

        n_critics = 2
        hidden_size = self.config.hidden_size
        batch_size, seq_length, n_features = x.shape

        for t in range(seq_length):
            h = nn.tanh(nn.Dense(hidden_size)(jnp.concatenate([x[:, t, :], jnp.expand_dims(h, axis=0)], axis=-1))).squeeze()
        critics = nn.Dense(n_critics)(h)

        return critics.squeeze(), h


# class PGCritic(nn.Module):
#     config: dict
#
#     @nn.compact
#     def __call__(self, x, h):
#
#         n_critics = 2
#         activation = nn.tanh
#
#         x = jnp.take(x, -1, axis=1)
#
#         critics = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         critics = activation(critics)
#         critics = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critics)
#         critics = activation(critics)
#         critics = nn.Dense(n_critics, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critics)
#
#         return critics.squeeze(), h

