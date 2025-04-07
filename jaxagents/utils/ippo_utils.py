from flax.training.train_state import TrainState
from flax import struct
from optax._src import base
import flax.linen
from gymnax.wrappers.purerl import LogEnvState
import numpy as np
from numpy.typing import NDArray
from typing import Dict, NamedTuple, Callable, Optional, Any, List, Annotated
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
import os


class Transition(NamedTuple):
    """Template for step transition"""
    """Environment state"""
    obs: Float[Array, "obs_size"]

    """Action selecetd by agent"""
    action: Int[Array, "n_actors"] | Float[Array, "n_actors"]

    """Value of the state"""
    value: Float[Array, "n_agents"]

    """Log-probability of selected policy action"""
    log_prob: Float[Array, "n_agents"]

    """Collected reward"""
    reward: Float[Array, "n_agents"]

    """Next environment state"""
    next_obs: Float[Array, "obs_size"]

    """Boolean variable indicating episode termination"""
    terminated: Bool[Array, "1"]

    """Value of next state"""
    next_value: Optional[Float[Array, "n_agents"]] = None

    """Advantage of step"""
    advantage: Optional[Float[Array, "n_agents"]] = None


class OptimizerParams(NamedTuple):
    """Parameters of the training optimizer"""
    """Learning rate"""
    learning_rate: float | Float[Array, "n_hyperparam_sets"] = 1e-3

    """Epsilon of the optimizer"""
    eps: float | Float[Array, "n_hyperparam_sets"] = 1e-8

    """Maximum value for gradient clipping"""
    grad_clip: float | Float[Array, "n_hyperparam_sets"] = 10.


class HyperParameters(NamedTuple):
    """Training hyperparameters for the DQN and DDQN agents"""
    """Gamma (discount parameter) of Bellman equation"""
    gamma: float | Float[Array, "n_hyperparam_sets"]

    """λ for weighting the discounted returns and the value as estimated by the critic in evaluating returns."""
    gae_lambda: float | Float[Array, "n_hyperparam_sets"]

    """Epsilon for policy ratio clipping"""
    eps_clip: float | Float[Array, "n_hyperparam_sets"]

    """Entropy coefficient for actor loss function"""
    ent_coeff: float | Float[Array, "n_hyperparam_sets"]

    """KL divergence threshold for early stopping of the actor training"""
    kl_threshold: float | Float[Array, "n_hyperparam_sets"]

    """Optimizer parameters for the actor network"""
    actor_optimizer_params: OptimizerParams

    """Optimizer parameters for the critic network"""
    critic_optimizer_params: OptimizerParams

    """
    Value function coefficient.
    Not relevant for the VPG-REINFORCE agent but help in using the same optimizer parameters for both actor and critic
    training.
    """
    vf_coeff: float | Float[Array, "n_hyperparam_sets"] = 1.


@struct.dataclass
class Runner:
    """
    Object for running, passes training status, environment state and hyperparameters between policy update steps.
    The runner is directed to have batch_size environment (states) and PRNG's but only a single TrainState per the actor
    and critic.
    """

    """Training status (params, training step and optimizer) of the actor"""
    actor_training: Annotated[List[TrainState], "n_agents"]

    """Training status (params, training step and optimizer) of the critic"""
    critic_training: Annotated[List[TrainState], "n_agents"]

    """State of the environment"""
    envstate: LogEnvState

    """State of the environment in array"""
    obs: Float[Array, "obs_size"]

    """Random key, required for reproducibility of results and control of randomness"""
    rng: PRNGKeyArray

    """Training hyperparameters"""
    hyperparams: HyperParameters

    """The loss value of the actors"""
    actor_loss: Float[Array, "1"]

    """The loss value of the critics"""
    critic_loss: Float[Array, "1"]


class AgentConfig(NamedTuple):
    """Configuration of the IPPO training algorithm agents, passed at initialization of instance."""

    """
    Number of training steps (not episodes).
    In case of continuing training, this is the additional steps to be performed.
    """
    n_steps: int

    """Size of batch collected from buffer for updating the agent's networks"""
    batch_size: int

    """Size of the minibatch batch to be used in updating the agent's networks."""
    minibatch_size: int

    """Number of steps to be collected when sampling trajectories"""
    rollout_length: int

    """Epochs for actor training per update step"""
    actor_epochs: int

    """Epochs for critic training per update step"""
    critic_epochs: int

    """Architecture of the actor network"""
    actor_network: flax.linen.Module

    """Architecture of the critic network"""
    critic_network: flax.linen.Module

    """Optax optimizer to be used in training. Giving only the optimizer class allows for initializing within the 
    self.train method and eventually running multiple combinations of the optimizer parameters via jax.vmap.
    """
    optimizer: Callable[[Any], Optional[base.GradientTransformation]]

    """Frequency of evaluating the agent in update steps."""
    eval_frequency: int = 1

    """PRNG key for evaluation of agent performance during training (if 'None' evaluation isn't performed)"""
    eval_rng: Optional[PRNGKeyArray] = None

    """Number of evaluations to be performed per evaluation step"""
    n_evals: Optional[int] = None

    """Absolute path for checkpointing"""
    checkpoint_dir: Optional[str | os.PathLike] = None

    """Whether an agent should be restored from training checkpoints, for continuing training or deploying."""
    restore_agent: bool = False

    "Maximum steps per episode for flagging episode truncation"
    max_episode_steps: int = 9999


@struct.dataclass
class MetricStats:
    """
    Dataclass summarizing statistics of a metric sample connected to the agent's performance (collected during either
    training or evaluation). Post-processed metrics are provided upon instance initialization to support the use of
    struct.dataclass, which is required for using jax.vmap.
    """
    """Metric per episode"""
    episode_metric: Annotated[NDArray[np.float32], "size_metrics"] | Float[Array, "size_metrics"]

    """Sample average"""
    mean: Annotated[NDArray[np.float32], "n_batch"] | Float[Array, "n_batch"]

    """Sample variance"""
    var: Annotated[NDArray[np.float32], "n_batch"] | Float[Array, "n_batch"]

    """Sample standard deviation"""
    std: Annotated[NDArray[np.float32], "n_batch"] | Float[Array, "n_batch"]

    """Sample minimum"""
    min: Annotated[NDArray[np.float32], "n_batch"] | Float[Array, "n_batch"]

    """Sample maximum"""
    max: Annotated[NDArray[np.float32], "n_batch"] | Float[Array, "n_batch"]

    """Sample median"""
    median: Annotated[NDArray[np.float32], "n_batch"] | Float[Array, "n_batch"]

    """Whether the sample contains nan values"""
    has_nans: Annotated[NDArray[np.bool], "n_batch"] | Bool[Array, "n_batch"]

