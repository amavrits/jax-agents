import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.core import FrozenDict
from flax import struct
import optax
from optax._src import base
import flax.linen
from gymnax.wrappers.purerl import LogEnvState
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union, Optional
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from dataclasses import dataclass


# class TrainStateVPG(NamedTuple):
#     """ Training state"""
#
#     """Actor network function"""
#     actor_apply_fn: Callable
#
#     """Critic network function"""
#     critic_apply_fn: Callable
#
#     """Parameters of the actor network"""
#     actor_params: FrozenDict
#
#     """Parameters of the critic network"""
#     critic_params: FrozenDict
#
#     """Optimizer for the actor network"""
#     actor_tx: optax.chain
#
#     """Optimizer for the critic network"""
#     critic_tx: optax.chain


class Transition(NamedTuple):
    """Template for step transition"""
    """Environment state"""
    state: Float[Array, "state_size"]

    """Action selecetd by agent"""
    action: Int[Array, "1"]

    """Value of the selected action"""
    value: Float[Array, "1"]

    """Log-probability of selected policy action"""
    log_prob: Float[Array, "1"]

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

    """Generalized Advantage Estimation lambda"""
    gae_lambda: float

    """Value clip"""
    clip_eps: float

    """ ??? """
    vf_coeff: float

    """ ??? """
    ent_coeff: float

    """Optimizer parameters for the actor network"""
    actor_optimizer_params: OptimizerParams

    """Optimizer parameters for the critic network"""
    critic_optimizer_params: OptimizerParams


@struct.dataclass
class Runner:
    """Object for running, passes training status, environment state and hyperparameters between training steps."""
    """Training status (params, training step and optimizer) of the actor"""
    actor_training: TrainState

    """Training status (params, training step and optimizer) of the critic"""
    critic_training: TrainState

    """State of the environment"""
    env_state: LogEnvState

    """State of the environment in array"""
    state: Float[Array, "state_size"]

    """Random key, required for reproducibility of results and control of randomness"""
    rng: PRNGKeyArray

    """Training hyperparameters"""
    hyperparams: HyperParameters


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

    """Size of batch collected from buffer for updating the policy network"""
    batch_size: int

    """Number of steps to be collected when sampling trajectories (must be large enough to sample entire batch)"""
    rollout_length: int

    """Number of epochs per policy update"""
    update_epochs: int

    """The architecture of the actor network"""
    actor_network: Type[flax.linen.Module]

    """The architecture of the critic network"""
    critic_network: Type[flax.linen.Module]

    """Template of transition, so that the buffer can be configured"""
    transition_template: Transition

    """Optax optimizer to be used in training. Giving only the optimizer class allows for initializing within the 
    self.train method and eventually running multiple combinations of the optimizer parameters via jax.vmap.
    """
    optimizer: Callable[[Dict], Optional[base.GradientTransformation]]

    """Type of loss function, not required for the Categorical DQN and QRDQN agents."""
    loss_fn: Optional[Callable[[Float[Array, "batch_size"], Float[Array, "batch_size"]], Float[Array, "1"]]] = None

    """Optional function for assessing the agent's performance during training."""
    get_performance: Optional[Callable[[int, Runner], Any]] = None

    """Optional function for defining the selection of random actions by the agent. This can be used to avoid illegal 
    actions and penalizing in the environment."""
    act_randomly: Callable[[PRNGKeyArray, Float[Array, "state_size"], int], int] =\
        lambda rng, state, n_actions: jax.random.choice(rng, jnp.arange(n_actions))

    """Whether the parameters and the performance of the agent should be stored during training."""
    store_agent: bool = False


@dataclass
class MetricStats:
    """
    Dataclass summarizing statistics of a metric sample connected to the agent's performance (collected during either
    training or evaluation).
    """
    """Metric per episode"""
    episode_metric: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

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
