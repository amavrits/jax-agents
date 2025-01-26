import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
from gymnax.environments import environment, spaces
from jaxtyping import Array, Int, Float
from typing import Optional
import chex
from typing import Tuple
import matplotlib.pyplot as plt


STATE = Float[Array, "n_predators+n_prey 2"]
ACTIONS = Int[Array, "n_predators+n_prey"]
REWARDS = Float[Array, "n_predators+n_prey"]


@struct.dataclass
class EnvState:
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


class Hunting(environment.Environment):

    def __init__(self) -> None:
        pass

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def get_obs(self, state: EnvState) -> STATE:
        return jnp.array(state.positions, dtype=jnp.float32).squeeze()

    def reset(self, key: chex.PRNGKey, params: Optional[EnvParams] = None) -> Tuple[STATE, EnvState]:

        self.time = 0.

        n_dim = env_params.n_predators + env_params.n_prey

        rng, *rngs = jax.random.split(key, 3)
        rng_x, rng_y = rngs

        x_coords = jax.random.uniform(rng_x, minval=min(env_params.x_lims), maxval=max(env_params.x_lims), shape=(n_dim,))
        y_coords = jax.random.uniform(rng_y, minval=min(env_params.y_lims), maxval=max(env_params.y_lims), shape=(n_dim,))
        state = jnp.stack((x_coords.T, y_coords.T), axis=-1)

        state = EnvState(positions=state)

        return self.get_obs(state), state

    def step(self, key: chex.PRNGKey, state: STATE, action: ACTIONS, params: EnvParams)\
            -> Tuple[STATE, EnvState, REWARDS, bool, bool, dict]:

        self.time += env_params.dt

        velocities = jnp.stack((
            jnp.repeat(env_params.prey_velocity, env_params.n_prey),
            jnp.repeat(env_params.predator_velocity, env_params.n_predators)
        ), axis=0)

        cond_list = [actions==0, actions==1, actions==2, actions==3]
        choice_list = [0, jnp.pi/2, jnp.pi, 3/2*jnp.pi]
        directions_rad = jnp.select(cond_list, choice_list)
        directions = jnp.stack((jnp.cos(directions_rad), jnp.sin(directions_rad)), -1)
        displacement =  env_params.dt * velocities * directions
        next_state = self.get_obs(state) + displacement
        next_state = jnp.stack((
            jnp.clip(jnp.take(next_state, 0, axis=1), min(env_params.x_lims), max(env_params.x_lims)),
            jnp.clip(jnp.take(next_state, 1, axis=1), min(env_params.y_lims), max(env_params.y_lims))
        ), axis=1)

        distance = jnp.linalg.norm((jnp.take(next_state, 0, axis=0)-jnp.take(next_state, 1, axis=0)))
        prey_caught = jnp.less_equal(distance, env_params.predator_radius)

        reward_prey = jnp.where(jnp.greater(distance, env_params.predator_radius), distance, -env_params.caught_reward)
        reward_predator = -reward_prey
        rewards = jnp.stack((reward_prey, reward_predator), axis=-1).squeeze()

        terminated = prey_caught
        truncated = self.time > env_params.max_time
        info = {"truncated": truncated}

        next_state = EnvState(positions=next_state)

        return (lax.stop_gradient(self.get_obs(next_state)),
                lax.stop_gradient(next_state),
                rewards.squeeze(),
                terminated.squeeze(),
                info)


    def render(self, state: STATE, actions: ACTIONS, env_params: EnvParams) -> plt.Figure:

        xticks = jnp.round(jnp.linspace(min(env_params.x_lims), max(env_params.x_lims), 6), 1)
        yticks = jnp.round(jnp.linspace(min(env_params.y_lims), max(env_params.y_lims), 6), 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(state[0, 0], state[0, 1], c="b", s=40, label="Prey")
        ax.scatter(state[1, 0], state[1, 1], c="r", s=80, label="Predator")
        pred_circle = plt.Circle((state[1, 0], state[1, 1]), env_params.predator_radius, color='r', alpha=0.3)
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

    @property
    def name(self) -> str:
        return "Hunting"

    def num_actions(self, params: EnvParams) -> int:
        return 4

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions(EnvParams))


if __name__ == "__main__":

    with jax.disable_jit(True):

        env_params = EnvParams()
        env = Hunting()

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
            actions = jax.random.choice(rng_action, jnp.arange(env.num_actions(env_params)), shape=(env_params.n_prey+env_params.n_predators,))
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

