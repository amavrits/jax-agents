import jax
import jax.numpy as jnp

from jax import lax
from jax_tqdm import scan_tqdm

from flax.training.train_state import TrainState
from flax.core import FrozenDict
from flax import struct
import flax.linen

import optax
import distrax
import flashbax as fbx

from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper, LogEnvState

from typing import Tuple, Dict, NamedTuple, Callable, Any
from abc import abstractmethod
from functools import partial


class TrainState(TrainState):
    target_params: FrozenDict


class OptimizerParams(NamedTuple):
    lr: jnp.float32 = 1e-3
    eps: jnp.float32 = 1e-3
    grad_clip: jnp.float32 = 1


class HyperParameters(NamedTuple):
    gamma: jnp.float32
    target_update_param: jnp.float32
    optimizer_params: OptimizerParams


class Transition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    terminated: jnp.ndarray
    info: Dict


@struct.dataclass
class Runner:
    training: TrainState
    env_state: LogEnvState
    state: jnp.array
    rng: jax.random.PRNGKey
    buffer_state: fbx.trajectory_buffer.BufferState
    hyperparams: HyperParameters


class AgentConfig(NamedTuple):
    n_steps: int
    buffer_size: int
    batch_size: int
    q_network: flax.linen.Module
    transition_template: Transition
    loss_fn: Callable[[jnp.array, jnp.array], jnp.array]
    set_optimizer: Callable[[OptimizerParams], optax.chain]
    get_performance: Callable[[int, Tuple], Any] = lambda i_step, runner: 0
    act_randomly: Callable[[jax.random.PRNGKey, jnp.array], int] = lambda rng, state: jax.random.choice(rng, jnp.arange(env.action_space(env_params).n)),
    buffer_type: str = "FLAT"
    target_update_method: str = "PERIODIC"
    epsilon_type: str = "DECAY"
    epsilon_params: Tuple = (0.9, 0.05, 50_000)
    store_agent: bool = False


class QLearningAgentBase:

    def __init__(self, env, env_params: NamedTuple, config: AgentConfig):
        self.config = config
        self._init_env(env, env_params)
        self._init_eps_fn(self.config.epsilon_type, self.config.epsilon_params)

    def _init_eps_fn(self, epsilon_type: str, epsilon_params: tuple):
        if epsilon_type == "CONSTANT":
            self.get_epsilon = jax.jit(lambda i_step: epsilon_params[0])
        elif epsilon_type == "DECAY":
            eps_start, eps_end, eps_decay = epsilon_params
            self.get_epsilon = jax.jit(lambda i_step: eps_end + (eps_start - eps_end) * jnp.exp(-i_step / eps_decay))

    def _init_env(self, env, env_params: NamedTuple):
        env = FlattenObservationWrapper(env)
        self.env = LogWrapper(env)
        self.env_params = env_params

    @partial(jax.jit, static_argnums=(0,))
    def _init_buffer(self):
        if self.config.buffer_type == "FLAT":
            self.buffer_fn = fbx.make_flat_buffer(
                max_length=self.config.buffer_size,
                min_length=self.config.batch_size,
                sample_batch_size=self.config.batch_size,
                add_sequences=True,
                add_batch_size=None,
            )
        elif self.config.buffer_type == "PER":
            raise Exception("PER buffers have not been added yet.")
        else:
            raise Exception("Unknown buffer type")

        buffer_state = self.buffer_fn.init(self.config.transition_template)

        return buffer_state

    def _init_optimizer(self, optimizer_params: OptimizerParams) -> optax.chain:
        return self.config.set_optimizer(optimizer_params)

    @partial(jax.jit, static_argnums=(0,))
    def _init_q_network(self, rng: jax.random.PRNGKey) -> Tuple[jax.random.PRNGKey, jnp.array]:
        rng, dummy_reset_rng, network_init_rng = jax.random.split(rng, 3)
        self.q_network = self.config.q_network(self.env.action_space(self.env_params).n, self.config)
        dummy_state, _ = self.env.reset(dummy_reset_rng, self.env_params)
        init_x = jnp.zeros((1, dummy_state.size))
        network_params = self.q_network.init(network_init_rng, init_x)
        return rng, network_params

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, rng: jax.random.PRNGKey) -> Tuple[jax.random.PRNGKey, jnp.array, LogEnvState]:
        rng, reset_rng = jax.random.split(rng)
        state, env_state = self.env.reset(reset_rng, self.env_params)
        return rng, state, env_state

    @partial(jax.jit, static_argnums=(0,))
    def _env_step(self, rng: jax.random.PRNGKey, env_state: NamedTuple, action: int) -> Tuple[jax.random.PRNGKey, jnp.array, LogEnvState, float, bool, Dict]:
        rng, step_rng = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info = self.env.step(step_rng, env_state, action.squeeze(), self.env_params)
        return rng, next_state, next_env_state, reward, terminated, info

    @partial(jax.jit, static_argnums=(0,))
    def _make_transition(self, state: jnp.array, action: int, reward: float, next_state: jnp.array, terminated: bool, info: Dict) -> Transition:
        transition = Transition(state.squeeze(), action, jnp.expand_dims(reward, axis=0), next_state,
                                jnp.expand_dims(terminated, axis=0), info)
        transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), transition)
        return transition

    @partial(jax.jit, static_argnums=(0,))
    def _store_transition(self, buffer_state: fbx.trajectory_buffer.BufferState, transition: Transition) -> fbx.trajectory_buffer.BufferState:
        return self.buffer_fn.add(buffer_state, transition)

    @partial(jax.jit, static_argnums=(0,))
    def _select_action(self, rng: jax.random.PRNGKey, state: jnp.array, training: TrainState, i_step: int) -> Tuple[jax.random.PRNGKey, int]:

        rng, *_rng = jax.random.split(rng, 3)
        random_action_rng, random_number_rng = _rng

        q_state = self._q(lax.stop_gradient(training.params), state)
        policy_action = jnp.argmax(q_state, 1)

        random_action = self.config.act_randomly(random_action_rng, state)

        random_number = jax.random.uniform(random_number_rng, minval=0, maxval=1, shape=(1,))
        eps = self.get_epsilon(i_step)
        exploitation = jnp.greater(random_number, eps)

        action = jnp.where(exploitation, policy_action, random_action)

        return rng, action

    @partial(jax.jit, static_argnums=(0,))
    def _update_target_network(self, runner: Tuple) -> Tuple:

        if self.config.target_update_method == "PERIODIC":

            training = runner.training.replace(target_params=optax.periodic_update(
                runner.training.params,
                runner.training.target_params,
                runner.training.step,
                runner.hyperparams.target_update_param
            ))

        elif self.config.target_update_method == "INCREMENTAL":

            training = runner.training.replace(target_params=optax.incremental_update(
                runner.training.params,
                runner.training.target_params,
                runner.hyperparams.target_update_param
            ))

        runner = runner.replace(training=training)

        return runner

    @partial(jax.jit, static_argnums=(0))
    def _fake_update_network(self, runner: Runner) -> Runner:
        return runner

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, runner: Runner, i_step: int) -> Tuple[Runner, Dict]:

        rng, action = self._select_action(runner.rng, runner.state, runner.training, i_step)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, runner.env_state, action)

        transition = self._make_transition(runner.state, action, reward, next_state, terminated, info)
        buffer_state = self._store_transition(runner.buffer_state, transition)

        runner = runner.replace(rng=rng, state=next_state, env_state=next_env_state, buffer_state=buffer_state)

        runner = jax.lax.cond(self.buffer_fn.can_sample(buffer_state),
                              self._update_q_network,
                              self._fake_update_network,
                              runner)

        runner = self._update_target_network(runner)

        metric = self._make_metrics(runner, reward, terminated)

        return runner, metric

    @partial(jax.jit, static_argnums=(0,))
    def _make_metrics(self, runner: Runner, reward: float, terminated: bool) -> Dict:

        if self.config.store_agent:
            metric = {
                "done": terminated,
                "reward": reward,
                "performance": self.config.get_performance(runner.training.step, runner),
                "network_params": jax.tree_util.tree_leaves(runner.training.params)
            }
        else:
            metric = {
                "done": terminated,
                "reward": reward,
                "performance": self.config.get_performance(runner.training.step, runner)
            }

        return metric

    @partial(jax.jit, static_argnums=(0,))
    def _sample_batch(self, rng: jax.random.PRNGKey, buffer_state: fbx.trajectory_buffer.BufferState) -> Tuple[jax.random.PRNGKey, fbx.trajectory_buffer.BufferSample]:
        rng, batch_sample_rng = jax.random.split(rng)
        batch = self.buffer_fn.sample(buffer_state, batch_sample_rng)
        batch = batch.experience
        return rng, batch

    @partial(jax.jit, static_argnums=(0,))
    def _update_q_network(self, runner: Runner) -> Runner:

        rng, batch = self._sample_batch(runner.rng, runner.buffer_state)

        current_state, action, reward, next_state, terminated = (
            batch.first.state.squeeze(),
            batch.first.action.squeeze(),
            batch.first.reward.squeeze(),
            batch.second.state.squeeze(),
            batch.first.terminated.squeeze(),
        )

        grad_fn = jax.jax.jit(jax.grad(self._loss, has_aux=False, allow_int=True, argnums=0))
        grads = grad_fn(runner.training.params,
                        runner.training.target_params,
                        current_state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        runner.hyperparams.gamma
                        )

        training = runner.training.apply_gradients(grads=grads)

        runner = runner.replace(rng=rng, training=training)

        return runner

    @jax.block_until_ready
    @partial(jax.jit, static_argnums=(0,))
    def train(self, rng: jax.random.PRNGKey, hyperparams: HyperParameters) -> Dict[Runner, Dict]:

        rng, network_params = self._init_q_network(rng)

        tx = self._init_optimizer(hyperparams.optimizer_params)

        training = TrainState.create(apply_fn=self.q_network.apply,
                                     params=network_params,
                                     target_params=network_params,
                                     tx=tx)

        buffer_state = self._init_buffer()

        rng, state, env_state = self._reset(rng)

        rng, runner_rng = jax.random.split(rng)

        step_runner = Runner(training, env_state, state, runner_rng, buffer_state, hyperparams)

        step_runner, metrics = lax.scan(
            scan_tqdm(self.config.n_steps)(self._step),
            step_runner,
            jnp.arange(self.config.n_steps),
            self.config.n_steps
        )

        return {"runner": step_runner, "metrics": metrics}

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _q(self, params: Dict, state: jnp.array) -> jnp.array:
        pass

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _q_state_action(self, params: Dict, state: jnp.array, action: int) -> jnp.array:
        pass

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _q_target(self, params: Dict, target_params: Dict, next_state: jnp.array, reward: float, terminated: bool, gamma: float) -> jnp.array:
        pass

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _loss(self, params: Dict, target_params: Dict, current_state: jnp.array, action: int, reward: float, next_state: jnp.array, terminated: bool, gamma: float) -> jnp.array:
        pass


class DQN_Agent(QLearningAgentBase):

    @partial(jax.jit, static_argnums=(0,))
    def _q(self, params: Dict, state: jnp.array) -> jnp.array:
        q_state = self.q_network.apply(params, state)
        return q_state

    @partial(jax.jit, static_argnums=(0,))
    def _q_state_action(self, params: Dict, state: jnp.array, action: int) -> float:
        action_batch_one_hot = jax.nn.one_hot(action.squeeze(), num_classes=self.env.action_space(self.env_params).n)
        q_state = self._q(params, state)
        q_state_action = jnp.sum(q_state * action_batch_one_hot, axis=1)
        return q_state_action

    @partial(jax.jit, static_argnums=(0,))
    def _q_target(self, params: Dict, target_params: Dict, next_state: jnp.array, reward: float, terminated: bool, gamma: float) -> jnp.array:
        q_target_next_state = self._q(lax.stop_gradient(target_params), next_state)
        q_target = reward.squeeze() + gamma * jnp.max(q_target_next_state, axis=1).squeeze() * jnp.logical_not(terminated.squeeze())
        return q_target

    @partial(jax.jit, static_argnums=(0,))
    def _loss(self, params: Dict, target_params: Dict, current_state: jnp.array, action: int, reward: float, next_state: jnp.array, terminated: bool, gamma: float) -> float:
        q_state_action = self._q_state_action(params, current_state, action)
        q_target = self._q_target(params, target_params, next_state, reward, terminated, gamma)
        loss = jnp.mean(jax.vmap(self.config["LOSS_FN"])(q_state_action, q_target), axis=0)
        return loss


class DDQN_Agent(QLearningAgentBase):

    @partial(jax.jit, static_argnums=(0,))
    def _q(self, params: Dict, state: jnp.array) -> jnp.array:
        q_state = self.q_network.apply(params, state)
        return q_state

    @partial(jax.jit, static_argnums=(0,))
    def _q_state_action(self, params: Dict, state: jnp.array, action: int) -> jnp.array:
        action_batch_one_hot = jax.nn.one_hot(action.squeeze(), num_classes=self.env.action_space(self.env_params).n)
        q_state = self._q(params, state)
        q_state_action = jnp.sum(q_state * action_batch_one_hot, axis=1)
        return q_state_action

    @partial(jax.jit, static_argnums=(0,))
    def _q_target(self, params: Dict, target_params: Dict, next_state: jnp.array, reward: float, terminated: bool, gamma: float) -> jnp.array:
        q_next_state = self._q(lax.stop_gradient(params), next_state)
        action_q_training = jnp.argmax(q_next_state, axis=1).reshape(-1, 1)
        q_target_next_state = jnp.take_along_axis(
            self._q(lax.stop_gradient(target_params), next_state),
            action_q_training,
            axis=-1).squeeze()
        q_target = reward.squeeze() + gamma * q_target_next_state * jnp.logical_not(terminated.squeeze())
        return q_target

    @partial(jax.jit, static_argnums=(0,))
    def _loss(self, params: Dict, target_params: Dict, current_state: jnp.array, action: int, reward: float, next_state: jnp.array, terminated: bool, gamma: float) -> jnp.array:
        q_state_action = self._q_state_action(params, current_state, action)
        q_target = self._q_target(params, target_params, next_state, reward, terminated, gamma)
        loss = jnp.mean(jax.vmap(self.config.loss_fn)(q_state_action, q_target), axis=0)
        return loss


class CategoricalDQN_Agent(QLearningAgentBase):

    @partial(jax.jit, static_argnums=(0,))
    def _q(self, params: dict, state: jnp.array) -> jnp.array:
        p_state = jax.nn.softmax(self.q_network.apply(params, state), axis=-1)
        q_state = jnp.dot(p_state, self.config["ATOMS"])
        return q_state

    @partial(jax.jit, static_argnums=(0,))
    def _q_state_action(self, params: Dict, state: jnp.array, action: int) -> jnp.array:
        action_batch_one_hot = jax.nn.one_hot(action.squeeze(), num_classes=self.env.action_space(self.env_params).n)
        logit_p_state = self.q_network.apply(params, state)
        logit_p_action_state = jnp.sum(logit_p_state * action_batch_one_hot[..., jnp.newaxis], axis=1)
        return logit_p_action_state

    @partial(jax.jit, static_argnums=(0,))
    def _q_target(self, params: Dict, target_params: Dict, next_state: jnp.array, reward: float, terminated: bool, gamma: float) -> jnp.array:

        q_next_state = self._q(lax.stop_gradient(params), next_state)
        action_next_state = jnp.argmax(q_next_state, axis=-1).squeeze()

        p_next_state = jax.nn.softmax(self.q_network.apply(lax.stop_gradient(target_params), next_state), axis=-1)
        p_next_state_action = jnp.take_along_axis(p_next_state, action_next_state[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze()

        T_Z = reward.reshape(-1, 1) + gamma * jnp.logical_not(terminated.reshape(-1, 1)) * self.config["ATOMS"]
        T_Z = jnp.clip(T_Z, self.config["ATOMS"].min(), self.config["ATOMS"].max())

        b_j = (T_Z - self.config["ATOMS"].min()) / self.config["DELTA_ATOMS"]
        lower = jnp.floor(b_j).astype(jnp.int32)
        upper = jnp.ceil(b_j).astype(jnp.int32)

        transform_lower = jax.nn.one_hot(lower, num_classes=self.config["ATOMS"].size).astype(jnp.int32)
        transform_upper = jax.nn.one_hot(upper, num_classes=self.config["ATOMS"].size).astype(jnp.int32)
        p_opt_upper = jnp.where(jnp.equal(jnp.remainder(b_j, 2), 0),
                                jnp.multiply(p_next_state_action, (upper - b_j)),
                                p_next_state_action * 0.5
                                )
        p_opt_lower = jnp.where(jnp.equal(jnp.remainder(b_j, 2), 0),
                                jnp.multiply(p_next_state_action, (b_j - lower)),
                                p_next_state_action * 0.5
                                )
        m_lower = jnp.multiply(p_opt_upper[..., jnp.newaxis], transform_lower).sum(axis=1)
        m_upper = jnp.multiply(p_opt_lower[..., jnp.newaxis], transform_upper).sum(axis=1)
        target = m_lower + m_upper

        return target

    @partial(jax.jit, static_argnums=(0,))
    def _cross_entropy(self, target: jnp.array, logit_p: jnp.array) -> jnp.array:
        return distrax.Categorical(probs=target).cross_entropy(distrax.Categorical(logits=logit_p))

    @partial(jax.jit, static_argnums=(0,))
    def _loss(self, params: Dict, target_params: Dict, current_state: jnp.array, action: int, reward: float, next_state: jnp.array, terminated: bool, gamma: float) -> jnp.array:
        logit_p_state_action = self._q_state_action(params, current_state, action)
        target = self._q_target(params, target_params, next_state, reward, terminated, gamma)
        loss = jnp.mean(jax.vmap(self._cross_entropy)(target, logit_p_state_action), axis=0)
        return loss


class QRDDQN_Agent(QLearningAgentBase):

    @partial(jax.jit, static_argnums=(0,))
    def _q(self, params: Dict, state: jnp.array) -> jnp.array:
        q_state = self.q_network.apply(params, state).mean(axis=-1)
        return q_state

    @partial(jax.jit, static_argnums=(0,))
    def _q_state_action(self, params: Dict, state: jnp.array, action: int) -> jnp.array:
        action_batch_one_hot = jax.nn.one_hot(action.squeeze(), num_classes=self.env.action_space(self.env_params).n)
        q_state = self.q_network.apply(params, state)
        q_state_action = jnp.sum(q_state * action_batch_one_hot[..., jnp.newaxis], axis=1)
        return q_state_action

    @partial(jax.jit, static_argnums=(0,))
    def _q_target(self, params: Dict, target_params: Dict, next_state: jnp.array, reward: float, terminated: bool, gamma: float) -> jnp.array:
        
        q_next_state = self._q(lax.stop_gradient(params), next_state)
        action_next_state = jnp.argmax(q_next_state, axis=-1).squeeze()

        q_next_state = self.q_network.apply(lax.stop_gradient(target_params), next_state)
        q_next_state_action = jnp.take_along_axis(q_next_state, action_next_state[:, jnp.newaxis, jnp.newaxis], axis=1).squeeze()
        q_target = reward.reshape(-1, 1) + gamma * q_next_state_action * jnp.logical_not(terminated.reshape(-1, 1))

        return q_target

    @partial(jax.jit, static_argnums=(0,))
    def _huber_loss(self, q: jnp.array, target: jnp.array) -> float:
        td_error = target[jnp.newaxis, :] - q[:, jnp.newaxis]
        huber_loss = jnp.where(
            jnp.less_equal(jnp.abs(td_error), self.config["HUBER_K"]),
            0.5 * td_error ** 2,
            self.config["HUBER_K"] * (jnp.abs(td_error) - 0.5 * self.config["HUBER_K"])
        )

        tau_hat = (jnp.arange(self.config["N_QUANTILES"], dtype=jnp.float32) + 0.5) / self.config["N_QUANTILES"]

        quantile_huber_loss = jnp.abs(tau_hat[:, jnp.newaxis] - jnp.less(td_error, 0).astype(jnp.int32)) * huber_loss
        quantile_huber_loss = jnp.where(
            jnp.logical_and(jnp.greater(q, -4), jnp.less(q, +4)),
            quantile_huber_loss,
            0
        )
        return jnp.sum(jnp.mean(quantile_huber_loss, 1), 0)
        
    @partial(jax.jit, static_argnums=(0,))
    def _loss(self, params: Dict, target_params: Dict, current_state: jnp.array, action: int, reward: float, next_state: jnp.array, terminated: bool, gamma: float) -> jnp.array:
        q_state_action = self._q_state_action(params, current_state, action)
        q_target = self._q_target(params, target_params, next_state, reward, terminated, gamma)
        loss = jnp.mean(jax.vmap(self._huber_loss)(q_target, q_state_action), axis=0)
        return loss


if __name__ == "__main__":

    import time
    import gymnax
    from cartpole_nn_gallery import *
    from postprocessing import PostProcessor

    env, env_params = gymnax.make("CartPole-v1")

    TRANSITION_TEMPLATE = Transition(
        state=jnp.zeros((1, 4), dtype=jnp.float32),
        action=jnp.zeros(1, dtype=jnp.int32),
        reward=jnp.zeros(1, dtype=jnp.float32),
        next_state=jnp.zeros((1, 4), dtype=jnp.float32),
        terminated=jnp.zeros(1, dtype=jnp.bool_),
        info={
            "discount": jnp.array((), dtype=jnp.float32),
            "returned_episode": jnp.array((), dtype=jnp.bool_),
            "returned_episode_lengths": jnp.array((), dtype=jnp.int32),
            "returned_episode_returns": jnp.array((), dtype=jnp.float32),
        }
    )

    def optimizer_fn(optimizer_params):
        return optax.chain(
            optax.clip_by_global_norm(optimizer_params.grad_clip),
            optax.rmsprop(learning_rate=optimizer_params.lr, eps=optimizer_params.eps)
            )


    n_atoms = 201
    n_quantiles = 51
    min_val, max_val = 0, 200
    DELTA_ATOMS = (max_val - min_val) / (n_atoms - 1)
    z = min_val + DELTA_ATOMS * jnp.arange(n_atoms)
    tau_hat = (jnp.arange(n_quantiles, dtype=jnp.float32) + 0.5) / n_quantiles
    pmf = jnp.ones(n_quantiles) / n_quantiles


    config = AgentConfig(
        q_network=DQN_NN_model,
        transition_template=TRANSITION_TEMPLATE,
        n_steps=500_000,
        buffer_type="FLAT",
        buffer_size=10_000,
        batch_size=128,
        target_update_method="PERIODIC",
        store_agent=False,
        act_randomly=lambda random_key, state: jax.random.choice(random_key, jnp.arange(env.action_space(env_params).n)),
        get_performance=lambda i_step, runner: 0,
        set_optimizer=optimizer_fn,
        loss_fn=optax.l2_loss,
        epsilon_type="DECAY",
        epsilon_params=(0.9, 0.05, 50_000)
    )

    # from jax.config import config as jconfig
    # jconfig.update('jax_disable_jax.jit', True)


    agent = DDQN_Agent(env, env_params, config)
    # agent = CategoricalDQN_Agent(config)
    # agent = QRDDQN_Agent(config)

    rng = jax.random.PRNGKey(42)
    t0 = time.time()

    hyperparams = HyperParameters(0.99, 4, OptimizerParams(5e-5, 0.01 / 32, 1))
    out = agent.train(rng, hyperparams)

    # gamma_grid = jnp.array([0.99])
    # target_update_grid = jnp.array([4])
    # lr_grid = jnp.array([1e-3, 1e-4])
    # eps_grid = jnp.array([0.01/32])
    # grad_clip_grid = jnp.array([1])
    # hyperparams = jnp.meshgrid(gamma_grid, target_update_grid, lr_grid, eps_grid, grad_clip_grid)
    # hyperparams = jnp.c_[[item.flatten() for item in hyperparams]].T
    # hyperparams = HyperParameters(hyperparams[0], hyperparams[1], OptimizerParams(hyperparams[2], hyperparams[3], hyperparams[4]))
    # vtrain = jax.vmap(agent.train, in_axes=(None, 0))
    # out = jax.block_until_ready(vtrain(rng, hyperparams))

    print(f"time: {time.time() - t0:.2f} s")


    pp = PostProcessor(out)
    fig = pp._plot_rewards(N=100)

