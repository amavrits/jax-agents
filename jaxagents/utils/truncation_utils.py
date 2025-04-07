import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.environments.classic_control.cartpole import CartPole
from typing import Tuple, Dict
from jaxtyping import Int, Float, Bool, Array


"""
IMPORTANT NOTE RESETTING TRUNCATION:
In order to truncation to behave as expected, the gymnax.environment.Environment parent must be able to reset within the
step() method. Also, the TruncationWrapper inherits gymnax.environment.Environment directly. Thus, the truncation
wrapper needs to call the reset_env() and step_env() methods of the environment, not the reset() and step() methods,
which would lead to the respective methods of the gymnax.environment.Environment inheritance of the environment. This is 
confusing, but many premade environments already inherit from gymnax, so the TruncationWrapper needs to work around in
this way.
"""


@struct.dataclass
class TruncationEnvState:
    envstate: EnvState
    step: int


class TruncationWrapper(Environment):
    def __init__(self, env: Environment, max_steps: int = 500):
        self._env = env
        self.max_steps = max_steps

    def reset_env(self, rng: jax.random.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, TruncationEnvState]:
        obs, envstate = self._env.reset_env(rng, params)
        return obs, TruncationEnvState(envstate=envstate, step=0)

    def step_env(
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
        next_obs, next_envstate, reward, terminated, info = self._env.step_env(rng, envstate.envstate, action, params)

        next_step = envstate.step + 1

        truncated = jnp.greater_equal(next_step, self.max_steps)
        done = jnp.logical_or(terminated, truncated)

        info = info.copy()
        info["terminated"] = terminated
        info["truncated"] = truncated

        next_envstate = TruncationEnvState(envstate=next_envstate, step=next_step)

        return next_obs, next_envstate, reward, done, info

