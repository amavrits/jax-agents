import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from gymnax.environments import environment, spaces
from jaxtyping import Array, Int, Float, Bool
from typing import Optional
import chex
from typing import Tuple
from abc import abstractmethod
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
import io


POSITIONS = Float[Array, "n_predators+n_prey 2"]
STATE = Float[Array, "1+n_predators+n_prey 2"]
ACTIONS = Int[Array, "n_predators+n_prey"]
DIRECTIONS = Int[Array, "n_predators+n_prey"]
VELOCITIES = Int[Array, "n_predators+n_prey"]
REWARDS = Float[Array, "n_predators+n_prey"]


@struct.dataclass
class EnvState:
    time: jnp.float32
    positions: STATE


@struct.dataclass
class EnvParams:
    n_prey: int = 1
    n_predators: int = 1
    prey_velocity: float = 1.
    predator_velocity: float = 1.
    predator_radius: float = .1
    max_time: float = 1.
    dt: float = 1e-2
    caught_reward: float = 10.
    x_lims: Tuple[float, float] = (0., 1.)
    y_lims: Tuple[float, float] = (0., 1.)


class HuntingBase(environment.Environment):

    def __init__(self, allow_timeover: bool = False):
        self.n_actors = 2
        self.allow_timeover = allow_timeover

    def get_obs(self, state: EnvState) -> STATE:
        return jnp.hstack((jnp.expand_dims(state.time, axis=-1), state.positions.reshape(1, -1)), dtype=jnp.float32)

    def reset_env(self, key: chex.PRNGKey, env_params: Optional[EnvParams] = None) -> Tuple[STATE, EnvState]:

        n_dim = 2  # FIXME: Make variable

        rng, *rngs = jax.random.split(key, 3)
        rng_x, rng_y = rngs

        x_coords = jax.random.uniform(rng_x, minval=env_params.x_lims[0], maxval=env_params.x_lims[1], shape=(n_dim,))
        y_coords = jax.random.uniform(rng_y, minval=env_params.y_lims[0], maxval=env_params.y_lims[1], shape=(n_dim,))

        state = jnp.stack((x_coords.T, y_coords.T), axis=-1)

        state = EnvState(time=jnp.zeros(1), positions=state)

        return self.get_obs(state), state

    def update_positions(self, positions:POSITIONS, directions: DIRECTIONS, velocities: VELOCITIES,
                         env_params: EnvParams) -> POSITIONS:

        displacement = env_params.dt * velocities * directions

        next_positions = positions + displacement

        next_positions = jnp.stack((
            jnp.clip(jnp.take(next_positions, 0, axis=1), env_params.x_lims[0], env_params.x_lims[1]),
            jnp.clip(jnp.take(next_positions, 1, axis=1), env_params.y_lims[0], env_params.y_lims[1])
        ), axis=1)

        return next_positions

    def _distance(self, positions: POSITIONS) -> Float[Array, "1"]:
        return jnp.linalg.norm((jnp.take(positions, 0, axis=0) - jnp.take(positions, 1, axis=0)))

    def _reward(self, distance: Float[Array, "1"], prey_caught: Bool[Array, "1"], timeover: Bool[Array, "1"],
                env_params: EnvParams) -> REWARDS:

        reward_prey = jnp.where(prey_caught, -env_params.caught_reward, distance)
        reward_predator = -reward_prey
        rewards = jnp.stack((reward_prey, reward_predator), axis=-1).squeeze()

        rewards_time_over = jnp.asarray([env_params.caught_reward, -env_params.caught_reward]).squeeze()
        rewards = jnp.where(timeover, rewards_time_over, rewards)

        return rewards

    def step_env(self, key: chex.PRNGKey, state: EnvState, actions: ACTIONS, env_params: EnvParams) \
            -> Tuple[STATE, EnvState, REWARDS, bool, bool, dict]:

        directions = self._directions(actions)

        velocities = jnp.expand_dims(jnp.asarray([env_params.prey_velocity, env_params.predator_velocity]), axis=-1)

        time, positions = state.time, state.positions
        next_time = state.time + env_params.dt
        next_positions = self.update_positions(positions, directions, velocities, env_params)
        next_state = EnvState(time=next_time, positions=next_positions)

        distance = self._distance(next_positions)

        prey_caught = jnp.less_equal(distance, env_params.predator_radius)
        timeover = jnp.greater(time, env_params.max_time) if self.allow_timeover else False
        terminated = jnp.logical_or(prey_caught, timeover)

        rewards = self._reward(distance, prey_caught, timeover, env_params)

        info = {
            "truncated": timeover,
            "rewards": rewards,
        }

        return (lax.stop_gradient(self.get_obs(next_state)),
                lax.stop_gradient(next_state),
                rewards.squeeze(),
                terminated.squeeze(),
                info)

    @abstractmethod
    def _directions(self, actions: ACTIONS) -> DIRECTIONS:
        raise NotImplementedError

    def render(self, env_state: STATE, actions: ACTIONS, env_params: EnvParams) -> plt.Figure:

        positions = env_state.squeeze()

        xticks = jnp.round(jnp.linspace(min(env_params.x_lims), max(env_params.x_lims), 6), 1)
        yticks = jnp.round(jnp.linspace(min(env_params.y_lims), max(env_params.y_lims), 6), 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(positions[0, 0], positions[0, 1], c="b", s=40, label="Prey")
        ax.scatter(positions[1, 0], positions[1, 1], c="r", s=80, label="Predator")
        pred_circle = plt.Circle((positions[1, 0], positions[1, 1]), env_params.predator_radius, color='r', alpha=0.3)
        ax.add_patch(pred_circle)
        ax.set_xlim(min(env_params.x_lims), max(env_params.x_lims))
        ax.set_ylim(min(env_params.y_lims), max(env_params.y_lims))
        ax.set_xticks(xticks, xticks)
        ax.set_yticks(yticks, yticks)
        ax.set_xlabel("X coordinate [m]", fontsize=12)
        ax.set_ylabel("Y coordinate [m]", fontsize=12)
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True)
        ax.grid()
        ax.set_aspect('equal', adjustable='box')
        plt.close()

        return fig

    def animate(self, states: STATE, actions: ACTIONS, env_params: EnvParams, gif_path: str) -> None:

        figs = list(map(lambda x: self.render(x, actions, env_params), states))

        image_frames = []
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            image_frames.append(img)
            plt.close(fig)

        image_frames[0].save(
            gif_path,
            save_all=True,
            append_images=image_frames[1:],
            duration=50,
            loop=0
        )

    @property
    def name(self) -> str:
        return "Hunting"



class HuntingDiscrete(HuntingBase):
    
    def _directions(self, actions: ACTIONS) -> DIRECTIONS:
        cond_list = [actions == 0, actions == 1, actions == 2, actions == 3]
        choice_list = [0, jnp.pi / 2, jnp.pi, 3 / 2 * jnp.pi]
        directions_rad = jnp.select(cond_list, choice_list)
        directions = jnp.stack((jnp.cos(directions_rad), jnp.sin(directions_rad)), -1)
        return directions


class HuntingContinuous(HuntingBase):

    def _directions(self, actions: ACTIONS) -> DIRECTIONS:
        return jnp.clip(actions, -jnp.pi, +jnp.pi)


if __name__ == "__main__":

    import numpy as np
    from jax_tqdm import scan_tqdm

    discrete = False

    env_params = EnvParams()

    env = HuntingDiscrete() if discrete else HuntingContinuous()

    def f(runner, i):
        rng, state, state_env = runner
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        actions = jax.random.normal(rng_action, shape=(env_params.n_prey+env_params.n_predators,))
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        runner = rng, next_state, next_env_state
        m = {
            "step": i,
            "state": state_env.positions.reshape(-1, 2, 2),
            "actions": actions,
            "next_state": next_env_state.positions.reshape(-1, 2, 2),
            "reward": reward,
            "terminated": terminated,
        }
        return runner, m

    n_eval_steps = 100
    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    eval_runner = rng, state, state_env
    eval_runner, metrics = jax.lax.scan(scan_tqdm(n_eval_steps)(f), eval_runner, jnp.arange(n_eval_steps))
    metrics = {key: np.asarray(val) for (key, val) in metrics.items()}

    gif_path = r"figures/random_discrete_policy.gif" if discrete else r"figures/random_continuous_policy.gif"
    env.animate(metrics["state"], [None, None], env_params, gif_path)

