import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from gymnax.environments import environment, spaces
from jaxtyping import Array, Int, Float
from typing import Optional
import chex
from typing import Tuple
from abc import abstractmethod
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
import io


STATE = Float[Array, "n_predators+n_prey 2"]
ACTIONS = Int[Array, "n_predators+n_prey"]
REWARDS = Float[Array, "n_predators+n_prey"]


@struct.dataclass
class EnvState:
    positions: STATE

@struct.dataclass
class EnvStateTime:
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


class HuntingDiscrete(environment.Environment):

    def reset_env(self, key: chex.PRNGKey, env_params: Optional[EnvParams] = None) -> Tuple[STATE, EnvState]:

        self.time = 0.

        n_dim = 2  # FIXME: Make variable

        rng, *rngs = jax.random.split(key, 3)
        rng_x, rng_y = rngs

        x_coords = jax.random.uniform(rng_x, minval=env_params.x_lims[0], maxval=env_params.x_lims[1], shape=(n_dim,))
        y_coords = jax.random.uniform(rng_y, minval=env_params.y_lims[0], maxval=env_params.y_lims[1], shape=(n_dim,))

        state = jnp.stack((x_coords.T, y_coords.T), axis=-1)

        state = EnvState(positions=state)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> STATE:
        return jnp.array(state.positions, dtype=jnp.float32).reshape(1, -1)

    def step_env(self, key: chex.PRNGKey, state: EnvStateTime, actions: ACTIONS, env_params: EnvParams) \
            -> Tuple[STATE, EnvStateTime, REWARDS, bool, dict]:

        self.time += env_params.dt

        velocities = jnp.stack((
            jnp.repeat(env_params.prey_velocity, 1),  # FIXME
            jnp.repeat(env_params.predator_velocity, 1)  # FIXME
        ), axis=0)

        cond_list = [actions == 0, actions == 1, actions == 2, actions == 3]
        choice_list = [0, jnp.pi / 2, jnp.pi, 3 / 2 * jnp.pi]
        directions_rad = jnp.select(cond_list, choice_list)
        directions = jnp.stack((jnp.cos(directions_rad), jnp.sin(directions_rad)), -1)
        displacement = env_params.dt * velocities * directions
        next_state = state.positions + displacement
        next_state = jnp.stack((
            jnp.clip(jnp.take(next_state, 0, axis=1), env_params.x_lims[0], env_params.x_lims[1]),
            jnp.clip(jnp.take(next_state, 1, axis=1), env_params.y_lims[0], env_params.y_lims[1])
        ), axis=1)

        distance = jnp.linalg.norm((jnp.take(next_state, 0, axis=0) - jnp.take(next_state, 1, axis=0)))
        prey_caught = jnp.less_equal(distance, env_params.predator_radius)

        reward_prey = jnp.where(jnp.greater(distance, env_params.predator_radius), distance, -env_params.caught_reward)
        reward_predator = -reward_prey
        rewards = jnp.stack((reward_prey, reward_predator), axis=-1).squeeze()

        terminated = prey_caught
        truncated = self.time > env_params.max_time
        info = {
            "truncated": truncated,
            "rewards": rewards,
        }

        next_state = EnvState(positions=next_state)

        return (lax.stop_gradient(self.get_obs(next_state)),
                lax.stop_gradient(next_state),
                rewards.squeeze(),
                terminated.squeeze(),
                info)

    def render(self, env_state: STATE, actions: ACTIONS, env_params: EnvParams) -> plt.Figure:

        # positions = env_state.positions.squeeze()
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

    def num_actions(self, env_params: EnvParams) -> int:
        return 4

    def action_space(self, env_params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions(env_params))


class HuntingContinuous(environment.Environment):

    def reset_env(self, key: chex.PRNGKey, env_params: Optional[EnvParams] = None) -> Tuple[STATE, EnvState]:

        n_dim = 2  # FIXME: Make variable

        rng, *rngs = jax.random.split(key, 3)
        rng_x, rng_y = rngs

        x_coords = jax.random.uniform(rng_x, minval=env_params.x_lims[0], maxval=env_params.x_lims[1], shape=(n_dim,))
        y_coords = jax.random.uniform(rng_y, minval=env_params.y_lims[0], maxval=env_params.y_lims[1], shape=(n_dim,))

        state = jnp.stack((x_coords.T, y_coords.T), axis=-1)

        state = EnvStateTime(time=jnp.zeros(1), positions=state)

        return self.get_obs(state), state

    def get_obs(self, state: EnvStateTime) -> STATE:
        # return jnp.hstack((jnp.expand_dims(state.time, axis=-1), state.positions.reshape(1, -1)), dtype=jnp.float32)
        return state.positions.reshape(1, -1)

    def step_env(self, key: chex.PRNGKey, state: EnvStateTime, actions: ACTIONS, env_params: EnvParams) \
            -> Tuple[STATE, EnvStateTime, REWARDS, bool, bool, dict]:

        # self.time += env_params.dt

        actions *= 2 * jnp.pi

        velocities = jnp.stack((
            jnp.repeat(env_params.prey_velocity, 1),  # FIXME
            jnp.repeat(env_params.predator_velocity, 1)  # FIXME
        ), axis=0)

        directions = jnp.stack((jnp.cos(actions), jnp.sin(actions)), -1)
        displacement = env_params.dt * velocities * directions
        next_positions = state.positions + displacement
        next_positions = jnp.stack((
            jnp.clip(jnp.take(next_positions, 0, axis=1), env_params.x_lims[0], env_params.x_lims[1]),
            jnp.clip(jnp.take(next_positions, 1, axis=1), env_params.y_lims[0], env_params.y_lims[1])
        ), axis=1)

        distance = jnp.linalg.norm((jnp.take(next_positions, 0, axis=0) - jnp.take(next_positions, 1, axis=0)))
        prey_caught = jnp.less_equal(distance, env_params.predator_radius)

        next_time = state.time + env_params.dt
        # time_over = jnp.greater(next_time, env_params.max_time)
        time_over = False
        terminated = jnp.logical_or(prey_caught, time_over)
        truncated = time_over

        reward_prey = jnp.where(jnp.greater(distance, env_params.predator_radius), distance, -env_params.caught_reward)
        reward_predator = -reward_prey
        rewards = jnp.stack((reward_prey, reward_predator), axis=-1).squeeze()

        rewards_time_over = jnp.asarray([env_params.caught_reward, -env_params.caught_reward]).squeeze()
        rewards = jnp.where(time_over, rewards_time_over, rewards)

        info = {
            "truncated": truncated,
            "rewards": rewards,
        }

        next_state = EnvStateTime(time=next_time, positions=next_positions)

        return (lax.stop_gradient(self.get_obs(next_state)),
                lax.stop_gradient(next_state),
                rewards.squeeze(),
                terminated.squeeze(),
                info)

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

    def num_actions(self, env_params: EnvParams) -> int:
        return 4

    def action_space(self, env_params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions(env_params))


if __name__ == "__main__":

    env_params = EnvParams()
    # env = HuntingDiscrete()
    env = HuntingContinuous()

    rng = jax.random.PRNGKey(42)
    rng, rng_reset = jax.random.split(rng, 2)
    state, env_state = env.reset(rng_reset, env_params)

    figs = []
    fig = env.render(state, jnp.zeros(2), env_params)
    figs.append(fig)

    step = 0
    done = False
    state_log = []
    action_log = []
    reward_log = []

    while not done:

        rng, rng_action, rng_step = jax.random.split(rng, 3)

        # Discrete
        # actions = jax.random.choice(rng_action, jnp.arange(env.num_actions(env_params)), shape=(env_params.n_prey+env_params.n_predators,))

        # Continuous
        actions = jax.random.beta(rng_action, a=2, b=2, shape=(env_params.n_prey+env_params.n_predators,))
        actions *= 2 * jnp.pi

        next_state, next_env_state, rewards, terminated, info = env.step(rng_step, env_state, actions, env_params)
        done = terminated or info["truncated"]
        fig = env.render(next_state, actions, env_params)
        figs.append(fig)
        step += 1
        state = next_state
        env_state = next_env_state

        state_log.append(state)
        action_log.append(actions)
        reward_log.append(rewards)

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(r"figures/random_policy_render.pdf")
    [pp.savefig(fig) for fig in figs]
    pp.close()



    import matplotlib.animation as animation
    from PIL import Image
    import io

    image_frames = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        image_frames.append(img)
        plt.close(fig)

    gif_path = r"figures/random_continuous_policy.gif"
    image_frames[0].save(
        gif_path,
        save_all=True,
        append_images=image_frames[1:],
        duration=50,  # Duration for each frame (in milliseconds)
        loop=0  # Loop 0 means infinite loop
    )