import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from flax import struct
import optax
import chex
import flashbax as fbx
import flax.linen
from gymnax.wrappers.purerl import LogEnvState
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union


class TrainStateDQN(TrainState):
    target_params: FrozenDict


class Transition(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    terminated: jnp.ndarray
    info: Dict


class OptimizerParams(NamedTuple):
    lr: Union[jnp.float32, jnp.ndarray] = 1e-3
    eps: Union[jnp.float32, jnp.ndarray] = 1e-3
    grad_clip: Union[jnp.float32, jnp.ndarray] = 1.0


class HyperParameters(NamedTuple):
    gamma: Union[jnp.float32, jnp.ndarray]
    target_update_param: Union[jnp.float32, jnp.ndarray]
    optimizer_params: OptimizerParams


class CategoricalHyperParameters(HyperParameters):
    pass


class QuantileHyperParameters(HyperParameters):
    huber_K: jnp.float32 = 1.0


@struct.dataclass
class Runner:
    training: TrainState
    env_state: LogEnvState
    state: jnp.ndarray
    rng: chex.PRNGKey
    buffer_state: fbx.trajectory_buffer.BufferState
    hyperparams: Union[HyperParameters, CategoricalHyperParameters, QuantileHyperParameters]


class AgentConfig(NamedTuple):
    n_steps: jnp.int32
    buffer_size: jnp.int32
    batch_size: jnp.int32
    q_network: Type[flax.linen.Module]
    transition_template: Transition
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    set_optimizer: Callable[[OptimizerParams], Type[optax.chain]]
    get_performance: Callable[[jnp.int32, Runner], Any] = lambda i_step, runner: 0
    act_randomly: Callable[[jax.Array, jnp.ndarray, jnp.int32], jnp.int32] =\
        lambda rng, state, n_actions: jax.random.choice(rng, jnp.arange(n_actions)),
    buffer_type: str = "FLAT"
    target_update_method: str = "PERIODIC"
    epsilon_type: str = "DECAY"
    epsilon_params: Tuple = (0.9, 0.05, 50_000)
    store_agent: jnp.bool_ = False


class CategoricalAgentConfig(AgentConfig):
    atoms: jnp.ndarray
    delta_atoms: jnp.ndarray


class QuantileAgentConfig(AgentConfig):
    n_qunatiles: jnp.int32 = 21

