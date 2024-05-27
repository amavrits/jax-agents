import pandas as pd
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import flashbax as fbx
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union, Optional
from agent_utils.dqn_datastructures import *


HyperParametersType = Union[HyperParameters, CategoricalHyperParameters, QuantileHyperParameters]
BufferStateType = fbx.trajectory_buffer.BufferState


@dataclass
class Stats:
    """
    Dataclass summarizing statistics of a metric sample connected to the agent's performance (collected during either
    training or evaluation).
    """
    """Sample average"""
    mean: np.float32
    """Sample variance"""
    var: np.float32
    """Sample standard deviation"""
    std: np.float32
    """Sample minimum"""
    min: np.float32
    """Sample maximum"""
    max: np.float32
    """Sample median"""
    median: np.float32
    """Whether the sample contains nan values"""
    has_nans: np.bool_


@struct.dataclass
class EvalRunner:
    """
    Runner for agent evaluation.
    """
    """State of the environment"""
    env_state: LogEnvState
    """State of the environment in array"""
    state: jnp.ndarray
    """Random key, required for reproducibility of results and control of randomness"""
    rng: chex.PRNGKey


class QAgentEvaluator:
    """
    Class used for postprocessing the agent training results, evaluating the agent's performance after training and
    using the agent to make operational policy suggestions.
    """
    """Whether the agent has been trained."""
    agent_trained: bool = False
    """Optimal policy network parameters after post-processing by parent class."""
    agent_params: Optional[FrozenDict] = None
    """Runner object after training."""
    training_runner: Optional[Runner] = None
    """Metrics collected during training."""
    training_metrics: Optional[Dict] = None


    def collect_train(self, runner: Runner = None, metrics: Dict = None) -> None:
        """Collects training of output (the final state of the runner after training and the collected metrics)."""
        self.agent_trained = True
        self.training_runner = runner
        self.training_metrics = metrics
        self._pp()


    def _pp(self) -> None:
        """
        Post-processes the training results,, which includes:
            - Setting the policy network parameters to be the parameters of the runner #TODO: Update for stored agent.
            - Extracting the buffer from the runner so that the training history can be exported.
        :return:
        """
        self.agent_params = self.training_runner.training.params
        self.buffer = self.training_runner.buffer_state


    @partial(jax.jit, static_argnums=(0,))
    def _eval_step(self, runner: EvalRunner, i_step: jnp.int32) -> Tuple[Runner, Dict]:
        """
        Performs an episode step for evaluation. This includes:
        - The agent selecting an action based on the trained policy network.
        - Performing an environment step using this action and the current state of the environment.
        - Generating metrics regarding the step.
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :param i_step: Current training step. Required for printing the progressbar via jax_tqdm.
        :return: A tuple containing:
                 - the step runner object, updated after performing an episode step.
                 - a dictionary of metrics regarding episode evolution and user-defined metrics.
        """

        q_values = self.q(runner.state)

        action = jnp.argmax(q_values)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(runner.rng, runner.env_state, action)

        """Update runner as a dataclass"""
        runner = runner.replace(env_state=next_env_state, state=next_state, rng=rng)

        metric = {"done": terminated, "reward": reward}

        return runner, metric


    def eval(self, rng: chex.PRNGKey, n_evals: int = 1e5) -> Dict:
        """
        Evaluates the trained agent's performance in the training environment. So, the performance of the agent can be
        isolated from agent training. The evaluation can be parallelized via jax.vmap.
        :param rng:  Random key for evaluation.
        :param n_evals: Number of steps in agent evaluation.
        :return: Dictionary of evaluation metrics.
        """

        rng, state, env_state = self._reset(rng)

        rng, runner_rng = jax.random.split(rng)

        runner = EvalRunner(env_state, state, rng)

        _, eval_metrics = lax.scan(
            scan_tqdm(n_evals)(self._eval_step),
            runner,
            jnp.arange(n_evals),
            n_evals
        )

        return eval_metrics


    @staticmethod
    def _stats(metric: np.ndarray) -> Stats:
        """
        Fills in the Stats dataclass with the summary statistics of the input metric.
        :param metric: The examined metric in numpy array format.
        :return: A Stats dataclass containing the summary statistics of the metric.
        """
        return Stats(
            mean=metric.mean(),
            var=metric.var(),
            std=metric.std(),
            min=metric.min(),
            max=metric.max(),
            median=np.median(metric),
            has_nans=np.isnan(metric)
        )

