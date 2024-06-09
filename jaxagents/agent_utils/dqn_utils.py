import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from flax import struct
from optax._src import base
import flashbax as fbx
import flax.linen
from gymnax.wrappers.purerl import LogEnvState
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union, Optional
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from dataclasses import dataclass


class TrainStateDQN(TrainState):
    """ Training state according to flax implementation"""
    """Parameters of the target network"""
    target_params: FrozenDict


class Transition(NamedTuple):
    """Template for step transition"""
    """Environment state"""
    state: Float[Array, "state_size"]

    """Action selecetd by agent"""
    action: Int[Array, "1"]

    """Collected reward"""
    reward: Float[Array, "1"]

    """Next environment state"""
    next_state: Float[Array, "state_size"]

    """Boolean variable indicating episode termination"""
    terminated: Bool[Array, "1"]

    """Dictionary of additional information about step"""
    info: Dict


class OptimizerParams(NamedTuple):
    """Parameters of the training optimizer"""
    """Learning rate"""
    learning_rate: float = 1e-3

    """Epsilon of the optimizer"""
    eps: float = 1e-3

    """Maximum value for gradient clipping"""
    grad_clip: float = 1.0


class HyperParameters(NamedTuple):
    """Training hyperparameters for the DQN and DDQN agents"""
    """Gamma (discount parameter) of Bellman equation"""
    gamma: float

    """Hyperparameter for updating the target network. Depending on updating style, it is either the period of updating
    or the rate of the incremental update."""
    target_update_param: float

    """Optimizer parameters"""
    optimizer_params: OptimizerParams


class CategoricalHyperParameters(HyperParameters):
    """Training hyperparameters for the Categorical DQN agent"""
    pass


class QuantileHyperParameters(HyperParameters):
    """Training hyperparameters for the QRDQN"""
    huber_K: float = 1.0


@struct.dataclass
class Runner:
    """Object for running, passes training status, environment state and hyperparameters between training steps."""
    """Training status (policy and traget network parameters), training step and optimizer"""
    training: TrainState

    """State of the environment"""
    env_state: LogEnvState

    """State of the environment in array"""
    state: Float[Array, "state_size"]

    """Random key, required for reproducibility of results and control of randomness"""
    rng: PRNGKeyArray

    """Training buffer"""
    buffer_state: fbx.trajectory_buffer.BufferState

    """Training hyperparameters"""
    hyperparams: Union[HyperParameters, CategoricalHyperParameters, QuantileHyperParameters]


@struct.dataclass
class EvalRunner:
    """
    Runner for agent evaluation.
    """
    """State of the environment"""
    env_state: LogEnvState

    """State of the environment in array"""
    state: Float[Array, "state_size"]

    """Random key, required for reproducibility of results and control of randomness"""
    rng: PRNGKeyArray


class AgentConfig(NamedTuple):
    """Configuration of the DQN and DDQN agents, passed at initialization of instance."""
    """Number of training steps (not episodes)"""
    n_steps: int

    """Size of the training buffer"""
    buffer_size: int

    """Size of batch collected from buffer for updating the policy network"""
    batch_size: int

    """The arcitecture of the policy (and target) network"""
    q_network: Type[flax.linen.Module]

    """Template of transition, so that the buffer can be configured"""
    transition_template: Transition

    """Optax optimizer to be used in training. Giving only the optimizer class allows for initializing within the 
    self.train method and eventually running multiple combinations of the optimizer parameters via jax.vmap.
    """
    optimizer: Optional[base.GradientTransformation] = None

    """Type of loss function, not required for the Categorical DQN and QRDQN agents."""
    loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None

    """Optional function for assessing the agent's performance during training."""
    get_performance: Optional[Callable[[int, Runner], Any]] = None

    """Optional function for defining the selection of random actions by the agent. This can be used to avoid illegal 
    actions and penalizing in the environment."""
    act_randomly: Callable[[jax.Array, jnp.ndarray, int], int] =\
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
    atoms: Float[Array, "n_atoms"]

    """Difference between atoms"""
    delta_atoms: Float[Array, "n_atoms"]


class QuantileAgentConfig(AgentConfig):
    """Configuration of the QRDQN agent, passed at initialization of instance."""
    """Number of quantiles"""
    n_qunatiles: int = 21


@dataclass
class MetricStats:
    """
    Dataclass summarizing statistics of a metric sample connected to the agent's performance (collected during either
    training or evaluation).
    """
    """Metric per episode"""
    episode_metric: Union[np.ndarray, Float[Array, "size_metrics"]]

    """Sample average"""
    mean: Union[np.float32, Float[Array, "size_metrics"]]

    """Sample variance"""
    var: Union[np.float32, Float[Array, "size_metrics"]]

    """Sample standard deviation"""
    std: Union[np.float32, Float[Array, "size_metrics"]]

    """Sample minimum"""
    min: Union[np.float32, Float[Array, "size_metrics"]]

    """Sample maximum"""
    max: Union[np.float32, Float[Array, "size_metrics"]]

    """Sample median"""
    median: Union[np.float32, Float[Array, "size_metrics"]]

    """Whether the sample contains nan values"""
    has_nans: Union[np.bool_, Bool[Array, "size_metrics"]]
