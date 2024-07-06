import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct
import optax
from optax._src import base
import flax.linen
from gymnax.wrappers.purerl import LogEnvState
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union, Optional
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from dataclasses import dataclass


class Transition(NamedTuple):
    """Template for step transition"""
    """Environment state"""
    state: Float[Array, "state_size"]

    """Action selecetd by agent"""
    action: Int[Array, "1"]

    """Value of the state"""
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

    """Value of next state"""
    next_value: Optional[Float[Array, "1"]] = None

    """Advantage of step"""
    advantage: Optional[Float[Array, "1"]] = None


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

    """λ for weighting the discounted returns and the value as estimated by the critic in evaluating returns."""
    gae_lambda: float

    """Epsilon for policy ratio clipping"""
    eps_clip: float

    """Entropy coefficient for actor loss function"""
    ent_coeff: float

    """
    Value function coefficient.
    Not relevant for the VPG-REINFORCE agent but help in using the same optimizer parameters for both actor and critic
    training.
    """
    vf_coeff: float

    """KL divergence threshold for early stopping of the actor training"""
    kl_threshold: float

    """Optimizer parameters for the actor network"""
    actor_optimizer_params: OptimizerParams

    """Optimizer parameters for the critic network"""
    critic_optimizer_params: OptimizerParams


@struct.dataclass
class Runner:
    """
    Object for running, passes training status, environment state and hyperparameters between policy update steps.
    The runner is directed to have batch_size environment (states) and PRNG's but only a single TrainState per the actor
    and critic.
    """
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


class AgentConfig(NamedTuple):
    """Configuration of the DQN and DDQN agents, passed at initialization of instance."""

    """Number of training steps (not episodes)"""
    n_steps: int

    """Size of batch collected from buffer for updating the policy network"""
    batch_size: int

    """Number of steps to be collected when sampling trajectories (must be large enough to sample entire batch)"""
    rollout_length: int

    """Architecture of the actor network"""
    actor_network: Type[flax.linen.Module]

    """Architecture of the critic network"""
    critic_network: Type[flax.linen.Module]

    """Epochs for actor training per update step"""
    actor_epochs: int

    """Epochs for critic training per update step"""
    critic_epochs: int

    """Optax optimizer to be used in training. Giving only the optimizer class allows for initializing within the 
    self.train method and eventually running multiple combinations of the optimizer parameters via jax.vmap.
    """
    optimizer: Callable[[Dict], Optional[base.GradientTransformation]]

    """PRNG key for evaluation of agent performance during training (if 'None' evaluation isn't performed)"""
    eval_rng: Optional[PRNGKeyArray] = None


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
