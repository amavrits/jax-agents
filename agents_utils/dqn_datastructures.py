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
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union, Optional


class TrainStateDQN(TrainState):
    """ Training state according to flax implementation"""
    """Parameters of the target network"""
    target_params: FrozenDict


class Transition(NamedTuple):
    """Template for step transition"""
    """Environment state"""
    state: jnp.ndarray
    """Action selecetd by agent"""
    action: jnp.ndarray
    """Collected reward"""
    reward: jnp.ndarray
    """Next environment state"""
    next_state: jnp.ndarray
    """Boolean variable indicating episode termination"""
    terminated: jnp.ndarray
    """Dictionary of additional information about step"""
    info: Dict


class OptimizerParams(NamedTuple):
    """Parameters of the training optimizer"""
    """Learning rate"""
    lr: Union[jnp.float32, jnp.ndarray] = 1e-3
    """Epsilon of the optimizer"""
    eps: Union[jnp.float32, jnp.ndarray] = 1e-3
    """Maximum value for gradient clipping"""
    grad_clip: Union[jnp.float32, jnp.ndarray] = 1.0


class HyperParameters(NamedTuple):
    """Training hyperparameters for the DQN and DDQN agents"""
    """Gamma (discount parameter) of Bellman equation"""
    gamma: Union[jnp.float32, jnp.ndarray]
    """Hyperparameter for updating the target network. Depending on updating style, it is either the period of updating
    or the rate of the incremental update."""
    target_update_param: Union[jnp.float32, jnp.ndarray]
    """Optimizer parameters"""
    optimizer_params: OptimizerParams


class CategoricalHyperParameters(HyperParameters):
    """Training hyperparameters for the Categorical DQN agent"""
    pass


class QuantileHyperParameters(HyperParameters):
    """Training hyperparameters for the QRDQN"""
    huber_K: jnp.float32 = 1.0


@struct.dataclass
class Runner:
    """Object for running, passes training status, environment state and hyperparameters between training steps."""
    """Training status (policy and traget network parameters), training step and optimizer"""
    training: TrainState
    """State of the environment"""
    env_state: LogEnvState
    """State of the environment in array"""
    state: jnp.ndarray
    """Random key, required for reproducibility of results and control of randomness"""
    rng: chex.PRNGKey
    """Training buffer"""
    buffer_state: fbx.trajectory_buffer.BufferState
    """Training hyperparameters"""
    hyperparams: Union[HyperParameters, CategoricalHyperParameters, QuantileHyperParameters]


class AgentConfig(NamedTuple):
    """Configuration of the DQN and DDQN agents, passed at initialization of instance."""
    """Number of training steps (not episodes)"""
    n_steps: jnp.int32
    """Size of the training buffer"""
    buffer_size: jnp.int32
    """Size of batch collected from buffer for updating the policy network"""
    batch_size: jnp.int32
    """The arcitecture of the policy (and target) network"""
    q_network: Type[flax.linen.Module]
    """Template of transition, so that the buffer can be configured"""
    transition_template: Transition
    """Function for setting the optimizer. Even though this approach is inconvenient, it allows for running multiple 
    combinations of the optimizer parameters via jax.vmap"""
    set_optimizer: Callable[[OptimizerParams], Type[optax.chain]]
    """Type of loss function, not required for the Categorical DQN and QRDQN agents."""
    loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    """Optional function for assessing the agent's performance during training."""
    get_performance: Optional[Callable[[jnp.int32, Runner], Any]] = None
    """Optional function for defining the selection of random actions by the agent. This can be used to avoid illegal 
    actions and penalizing in the environment."""
    act_randomly: Callable[[jax.Array, jnp.ndarray, jnp.int32], jnp.int32] =\
        lambda rng, state, n_actions: jax.random.choice(rng, jnp.arange(n_actions))
    """Type of the training buffer"""
    buffer_type: str = "FLAT"
    """Style of updating the target network parameters"""
    target_update_method: str = "PERIODIC"
    """Style of function for reducing the epsilon value of the epsilon-greedy policy."""
    epsilon_fn_style: str = "DECAY"
    """Parameters of function for reducing the epsilon value of the epsilon-greedy policy."""
    epsilon_params: Tuple = (0.9, 0.05, 50_000)
    """Whether the parameters and the performance of the agent should be stored during training."""
    store_agent: jnp.bool_ = False


class CategoricalAgentConfig(AgentConfig):
    """Configuration of the Categorical DQN agent, passed at initialization of instance."""
    """Atoms in an array. Passing this argument instead of the minimum and maximum values can increase the flexibility
    of applying the agent."""
    atoms: jnp.ndarray
    """Difference between atoms"""
    delta_atoms: jnp.ndarray


class QuantileAgentConfig(AgentConfig):
    """Configuration of the QRDQN agent, passed at initialization of instance."""
    """Number of quantiles"""
    n_qunatiles: jnp.int32 = 21

