import numpy as np
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax import struct
from optax._src import base
import flax.linen
from gymnax.wrappers.purerl import LogEnvState
from typing import Dict, NamedTuple, Callable, Type, Union, Optional, Any
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from dataclasses import dataclass, field
import os


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
    learning_rate: Union[float, Float[Array, "n_hyperparam_sets"]] = 1e-3

    """Epsilon of the optimizer"""
    eps: Union[float, Float[Array, "n_hyperparam_sets"]] = 1e-8

    """Maximum value for gradient clipping"""
    grad_clip: Union[float, Float[Array, "n_hyperparam_sets"]] = 10.0


class HyperParameters(NamedTuple):
    """Training hyperparameters for the DQN and DDQN agents"""
    """Gamma (discount parameter) of Bellman equation"""
    gamma: Union[float, Float[Array, "n_hyperparam_sets"]]

    """λ for weighting the discounted returns and the value as estimated by the critic in evaluating returns."""
    gae_lambda: Union[float, Float[Array, "n_hyperparam_sets"]]

    """Epsilon for policy ratio clipping"""
    eps_clip: Union[float, Float[Array, "n_hyperparam_sets"]]

    """Entropy coefficient for actor loss function"""
    ent_coeff: Union[float, Float[Array, "n_hyperparam_sets"]]

    """KL divergence threshold for early stopping of the actor training"""
    kl_threshold: Union[float, Float[Array, "n_hyperparam_sets"]]

    """Optimizer parameters for the actor network"""
    actor_optimizer_params: OptimizerParams

    """Optimizer parameters for the critic network"""
    critic_optimizer_params: OptimizerParams

    """
    Value function coefficient.
    Not relevant for the VPG-REINFORCE agent but help in using the same optimizer parameters for both actor and critic
    training.
    """
    vf_coeff: Union[float, Float[Array, "n_hyperparam_sets"]] = 1.0


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

    """The loss value of the actor"""
    actor_loss: Float[Array, "1"]

    """The loss value of the critic"""
    critic_loss: Float[Array, "1"]


class AgentConfig(NamedTuple):
    """Configuration of the PPO agents, passed at initialization of instance."""

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
    # optimizer: Callable[[Dict], Optional[base.GradientTransformation]]
    optimizer: Callable[[Any], Optional[base.GradientTransformation]]

    """Frequency of evaluating the agent in update steps."""
    eval_frequency: int = 1

    """PRNG key for evaluation of agent performance during training (if 'None' evaluation isn't performed)"""
    eval_rng: Optional[PRNGKeyArray] = None

    """Number of evaluations to be performed per evaluation step"""
    n_evals: Optional[int] = None

    """Absolute path for checkpointing"""
    checkpoint_dir: Optional[Union[str, os.PathLike]] = None

    """Whether an agent should be restored from training checkpoints, for continuing training or deploying."""
    restore_agent: bool = False


@struct.dataclass
class MetricStats:
    """
    Dataclass summarizing statistics of a metric sample connected to the agent's performance (collected during either
    training or evaluation). Post-processed metrics are provided upon instance initialization to support the use of
    struct.dataclass, which is required for using jax.vmap.
    """
    """Metric per episode"""
    episode_metric: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Sample average"""
    mean: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Sample variance"""
    var: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Sample standard deviation"""
    std: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Sample minimum"""
    min: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Sample maximum"""
    max: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Sample median"""
    median: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]

    """Whether the sample contains nan values"""
    has_nans: Union[np.ndarray["size_metrics", bool], Bool[Array, "size_metrics"]]
