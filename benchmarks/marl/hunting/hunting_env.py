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
from matplotlib.backends.backend_pdf import PdfPages

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
    directions: DIRECTIONS
    distance: jnp.float32
    action_mask: Float[Array, "2"]


@struct.dataclass
class EnvParams:
    n_prey: int = 1
    n_predators: int = 1
    prey_velocity: float = 1.
    predator_velocity: float = 1.
    predator_radius: float = .2
    max_time: float = 1.
    dt: float = .01
    caught_reward: float = 100.
    x_lims: Tuple[float, float] = (0., 1.)
    y_lims: Tuple[float, float] = (0., 1.)


class HuntingBase(environment.Environment):

    def __init__(self):
        self.n_actors = 2

    def get_obs(self, state: EnvState) -> STATE:
        return jnp.hstack((jnp.expand_dims(state.time, axis=-1), state.positions.reshape(1, -1), state.directions.reshape(1, -1)), dtype=jnp.float32)
        # return jnp.hstack((jnp.expand_dims(state.time, axis=-1), state.positions.reshape(1, -1)), dtype=jnp.float32)

    def reset_env(self, key: chex.PRNGKey, env_params: Optional[EnvParams] = None) -> Tuple[STATE, EnvState]:

        # key = jax.random.PRNGKey(11)

        rng, *rngs = jax.random.split(key, 3)
        rng_x, rng_y = rngs

        x_coords = jax.random.uniform(rng_x, minval=env_params.x_lims[0], maxval=env_params.x_lims[1], shape=(self.n_actors,))
        y_coords = jax.random.uniform(rng_y, minval=env_params.y_lims[0], maxval=env_params.y_lims[1], shape=(self.n_actors,))

        positions = jnp.stack((x_coords.T, y_coords.T), axis=-1)

        directions = jnp.zeros(2)

        distance = self._distance(positions)

        action_mask = jnp.asarray([-jnp.pi, +jnp.pi])

        state = EnvState(time=jnp.zeros(1), positions=positions, directions=directions, distance=distance, action_mask=action_mask)

        return self.get_obs(state), state

    @abstractmethod
    def _adjust_bnds(self, positions: POSITIONS, env_params: EnvParams):
        raise NotImplementedError

    def update_positions(self, positions:POSITIONS, directions: DIRECTIONS, velocities: VELOCITIES,
                         env_params: EnvParams) -> POSITIONS:

        displacement = env_params.dt * velocities * directions

        next_positions = positions + displacement

        next_positions = self._adjust_bnds(next_positions, env_params)

        return next_positions

    def _distance(self, positions: POSITIONS) -> Float[Array, "1"]:
        return jnp.linalg.norm(jnp.diff(positions, axis=0))

    def step_env(self, key: chex.PRNGKey, state: EnvState, actions: ACTIONS, env_params: EnvParams) \
            -> Tuple[STATE, EnvState, REWARDS, bool, bool, dict]:

        directions = self._directions(actions)

        velocities = jnp.expand_dims(jnp.asarray([env_params.prey_velocity, env_params.predator_velocity]), axis=-1)

        time, positions, distance = state.time, state.positions, state.distance
        next_time = time + env_params.dt
        next_positions = self.update_positions(positions, directions, velocities, env_params)
        next_distance = self._distance(next_positions)
        next_action_mask = jnp.asarray([-jnp.pi, +jnp.pi])
        next_directions = jnp.clip(actions, -jnp.pi, +jnp.pi)
        next_state = EnvState(time=next_time, positions=next_positions, directions=next_directions, distance=next_distance, action_mask=next_action_mask)

        prey_caught = jnp.less_equal(next_distance, env_params.predator_radius)
        truncated = jnp.greater_equal(time, env_params.max_time)  # Truncation = time over
        terminated = jnp.logical_or(prey_caught, truncated)

        movement = jnp.linalg.norm(next_positions-positions, axis=-1)
        eff_velocity = movement / env_params.dt
        # Avoid numerical inaccuracy of velocity --> stuck when effective velocity is 95% of maximum
        stuck = jnp.less(eff_velocity.squeeze(), velocities.squeeze()*0.95)

        step_rewards = jnp.asarray([+1, -1])
        caught_rewards = jnp.asarray([-1, +1]) * env_params.caught_reward
        rewards = step_rewards * (1 - prey_caught) + caught_rewards * prey_caught
        # rewards = step_rewards * (1 - prey_caught) + caught_rewards * prey_caught + jnp.asarray([1, 0]) * (1 - stuck)

        info = {
            "truncated": truncated.squeeze(),
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

    def render(self, time: Float[Array, "1"], positions: POSITIONS, actions: ACTIONS, values: REWARDS,
               env_params: EnvParams) -> plt.Figure:

        positions = positions.squeeze()
        actions = actions.squeeze()
        values = values.squeeze()
        directions = self._directions(actions)
        length = max(env_params.x_lims) / 10
        dx = length * jnp.take(directions, 0, axis=-1)
        dy = length * jnp.take(directions, 1, axis=-1)

        xticks = jnp.round(jnp.linspace(min(env_params.x_lims), max(env_params.x_lims), 6), 1)
        yticks = jnp.round(jnp.linspace(min(env_params.y_lims), max(env_params.y_lims), 6), 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle("Time={t:.2f}sec".format(t=time), fontsize=14)
        ax.scatter(positions[0, 0], positions[0, 1], c="b", s=40, label="Prey")
        ax.scatter(positions[1, 0], positions[1, 1], c="r", s=80, label="Predator")
        ax.arrow(positions[0, 0], positions[0, 1], dx[0], dy[0], color="b", linewidth=1)
        ax.arrow(positions[1, 0], positions[1, 1], dx[1], dy[1], color="r", linewidth=1)
        ax.annotate("${V}_{s}$"+"={value:.2f}".format(value=values[0]), (positions[0, 0], positions[0, 1]+0.01), color="b")
        ax.annotate("${V}_{s}$"+"={value:.2f}".format(value=values[1]), (positions[1, 0], positions[1, 1]+0.01), color="r")
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

    def animate(self, times: Float[Array, "n_steps"], positions: POSITIONS, actions: ACTIONS, values: REWARDS,
                env_params: EnvParams, gif_path: str, export_pdf: bool = False) -> None:

        figs = list(map(lambda w, x, y, z: self.render(w, x, y, z, env_params), times, positions, actions, values))

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

        if export_pdf:
            pdf_path = gif_path[:-3]+"pdf"
            pp = PdfPages(pdf_path)
            [pp.savefig(fig) for fig in figs]
            pp.close()

    @property
    def name(self) -> str:
        return "Hunting"

    def simple_strategy(self, env_state: EnvState, env_params: EnvParams) -> ACTIONS:
        positions = env_state.positions
        positions_diff = jnp.diff(positions, axis=0)
        angle = jnp.atan(jnp.take())
        return jnp.asarray([angle.squeeze(), angle.squeeze()])

class HuntingDiscrete(HuntingBase):
    
    def _directions(self, actions: ACTIONS) -> DIRECTIONS:
        cond_list = [actions == 0, actions == 1, actions == 2, actions == 3]
        choice_list = [0, jnp.pi / 2, jnp.pi, 3 / 2 * jnp.pi]
        directions_rad = jnp.select(cond_list, choice_list)
        directions = jnp.stack((jnp.cos(directions_rad), jnp.sin(directions_rad)), axis=-1)
        return directions

    def _adjust_bnds(self, positions: POSITIONS, env_params: EnvParams):
        positions = jnp.stack((
            jnp.clip(jnp.take(positions, 0, axis=1), env_params.x_lims[0], env_params.x_lims[1]),
            jnp.clip(jnp.take(positions, 1, axis=1), env_params.y_lims[0], env_params.y_lims[1])
        ), axis=1)
        return positions


class HuntingContinuous(HuntingBase):

    def _directions(self, actions: ACTIONS) -> DIRECTIONS:
        actions_clip = jnp.clip(actions, -jnp.pi, +jnp.pi)
        directions = jnp.stack((jnp.cos(actions_clip), jnp.sin(actions_clip)), axis=-1)
        return directions

    def _adjust_bnds(self, positions: POSITIONS, env_params: EnvParams):
        positions = jnp.stack((
            jnp.clip(jnp.take(positions, 0, axis=1), env_params.x_lims[0], env_params.x_lims[1]),
            jnp.clip(jnp.take(positions, 1, axis=1), env_params.y_lims[0], env_params.y_lims[1])
        ), axis=1)
        return positions


if __name__ == "__main__":

    import numpy as np
    from jax_tqdm import scan_tqdm

    discrete = True
    # discrete = False

    env_params = EnvParams(prey_velocity=2, predator_velocity=1)
    env = HuntingDiscrete() if discrete else HuntingContinuous()

    def f(runner, i):
        rng, state, state_env = runner
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        if discrete:
            actions = jax.random.choice(rng_action, jnp.arange(4), shape=(env.n_actors,))
        else:
            actions = jax.random.uniform(rng_action, minval=-1., maxval=+1., shape=(env.n_actors,))
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        runner = rng, next_state, next_env_state
        metrics = {
            "step": i,
            "positions": state_env.positions.reshape(-1, 2, 2),
            "actions": actions,
            "next_positions": next_env_state.positions.reshape(-1, 2, 2),
            "reward": reward,
            "terminated": terminated,
        }
        return runner, metrics

    n_eval_steps = 300
    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    render_runner = rng, state, state_env
    render_runner, render_metrics = jax.lax.scan(scan_tqdm(n_eval_steps)(f), render_runner, jnp.arange(n_eval_steps))
    render_metrics = {key: np.asarray(val) for (key, val) in render_metrics.items()}

    if discrete:
        gif_path = r"figures/discrete/random_discrete_policy.gif"
    else:
        gif_path = r"figures/continuous/random_continuous_policy.gif"
    env.animate(render_metrics["positions"], render_metrics["actions"], env_params, gif_path)

