import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax.numpy as jnp


class PGActorDiscrete(nn.Module):
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

        return actor_mean.reshape(n_actors, action_dim)


# class PGActorContinuous(nn.Module):
#     config: dict
#
#     @nn.compact
#     def __call__(self, x):
#
#         n_actors = 2
#         activation = nn.tanh
#
#         # actors = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         # actors = activation(actors)
#         actors = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         actors = activation(actors)
#         actors = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actors)
#         actors = activation(actors)
#         actors = nn.Dense(n_actors, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actors)
#
#         return actors
#
#
# class PGCritic(nn.Module):
#     config: dict
#
#     @nn.compact
#     def __call__(self, x):
#
#         n_critics = 2
#         activation = nn.tanh
#
#         critics = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
#         critics = activation(critics)
#         critics = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critics)
#         critics = activation(critics)
#         critics = nn.Dense(n_critics, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critics)
#
#         return critics.squeeze()



class PGActorContinuous(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        n_actors = 2
        activation = nn.tanh

        actors = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        actors = activation(actors)
        actors = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actors)
        actors = activation(actors)
        actors1 = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actors)

        actors = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        actors = activation(actors)
        actors = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(actors)
        actors = activation(actors)
        actors2 = nn.Dense(1, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actors)

        actors = jnp.stack((actors1, actors2))

        return actors.squeeze()


class PGCritic(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):

        n_critics = 2
        activation = nn.tanh

        critics = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        critics = activation(critics)
        critics = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critics)
        critics = activation(critics)
        critics1 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critics)

        critics = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        critics = activation(critics)
        critics = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critics)
        critics = activation(critics)
        critics2 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critics)

        critics = jnp.stack((critics1, critics2))

        return critics.squeeze()

