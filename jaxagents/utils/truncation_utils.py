from flax.struct import dataclass
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.wrappers.purerl import LogEnvState
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from typing import Tuple, Dict


@dataclass
class TruncationEnvState:
    envstate: EnvState | LogEnvState
    steps: int


class TruncationWrapper(Environment):

    def __init__(self, env: Environment, max_episode_steps: int = jnp.inf):
        self.env = env
        self.max_steps = max_episode_steps

    def reset(self, rng: PRNGKeyArray, params: EnvParams) -> Tuple[Float[Array, "obs_size"], TruncationEnvState]:
        obs, envstate = self.env.reset(rng, params)
        return obs, TruncationEnvState(envstate, steps=0)

    def step(self,
             rng: PRNGKeyArray,
             envstate: TruncationEnvState,
             action: Int[Array, "n_actors"] | Float[Array, "n_actors"],
             params: EnvParams
             ) -> Tuple[
        Float[Array, "obs_size"],
        TruncationEnvState,
        Float[Array, "n_agents"],
        Bool[Array, "1"],
        Dict[str, bool | float]
    ]:

        obs, next_envstate, reward, terminated, info = self.env.step(rng, envstate.envstate, action, params)

        steps = envstate.steps + 1
        truncated = jnp.greater_equal(steps, self.max_steps)
        info.update({"terminated": terminated})
        info.update({"truncated": truncated})

        done = jnp.logical_or(terminated, truncated)

        return (
            obs,
            TruncationEnvState(next_envstate, steps),
            reward,
            done,
            info
        )

