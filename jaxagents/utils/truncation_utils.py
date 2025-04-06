import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments.classic_control.cartpole import CartPole
from typing import Tuple, Dict
from jaxtyping import Int, Float, Bool, Array

@struct.dataclass
class TruncationEnvState:
    envstate: EnvState
    step: int


class TruncationWrapper(Environment):
    def __init__(self, env: Environment, max_steps: int = 500):
        self._env = env
        self.max_steps = max_steps

    def reset(self, rng: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, TruncationEnvState]:
        obs, envstate = self._env.reset(rng, params)
        return obs, TruncationEnvState(envstate=envstate, step=0)

    def step(
            self,
            rng: jax.random.PRNGKey,
            envstate: TruncationEnvState,
            action: Int[Array, "n_actors"] | Float[Array, "n_actors"],
            params: EnvParams
    ) -> Tuple[
        Float[Array, "state_size"],
        TruncationEnvState,
        Float[Array, "n_agents"],
        Bool[Array, "1"],
        Dict[str, float | bool]
    ]:

        # Termination is determined by whether the environment is done (no other info is available)
        next_obs, next_envstate, reward, terminated, info = self._env.step(rng, envstate.envstate, action, params)

        next_step = envstate.step + 1

        truncated = jnp.greater_equal(next_step, self.max_steps)
        done = jnp.logical_or(terminated, truncated)

        next_step = jnp.where(done, 0, next_step)

        info = info.copy()
        info["terminated"] = terminated
        info["truncated"] = truncated

        next_envstate = TruncationEnvState(envstate=next_envstate, step=next_step)

        return next_obs, next_envstate, reward, done, info

