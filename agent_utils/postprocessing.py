import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import flashbax as fbx
from dataclasses import dataclass
from typing import Tuple, Dict, NamedTuple, Callable, Any, Type, Union, Optional
from agent_utils.dqn_datastructures import *


HyperParametersType = Union[HyperParameters, CategoricalHyperParameters, QuantileHyperParameters]
BufferStateType = fbx.trajectory_buffer.BufferState

@dataclass
class Stats:
    mean: np.float32
    var: np.float32
    std: np.float32
    min: np.float32
    max: np.float32
    median: np.float32
    has_nans: np.bool_


class PostProcessor:
    agent_trained: bool = False
    agent_params: Optional[FrozenDict] = None
    training_runner: Optional[Runner] = None
    training_metrics: Optional[Dict] = None

    def collect_train(self, runner: Optional[Runner] = None, metrics: Optional[Dict] = None) -> None:
        self.agent_trained = True
        self.training_runner = runner
        self.training_metrics = metrics
        self._pp()

    def _pp(self) -> None:
        self.agent_params = self.training_runner.training.params
        self.buffer = self.training_runner.buffer_state
        self._collect_output()

    def _collect_output(self):
        self.dones = np.asarray(self.training_metrics["done"])
        self.rewards = np.asarray(self.training_metrics["reward"])
        self.reward_stats = self._stats(self.rewards)

    @staticmethod
    def _stats(metric: np.ndarray) -> Stats:
        return Stats(
            mean=metric.mean(),
            var=metric.var(),
            std=metric.std(),
            min=metric.min(),
            max=metric.max(),
            median=np.median(metric),
            has_nans=np.isnan(metric)
        )

    #TODO: Make property
    @staticmethod
    def get_running_metric(metric, running_window):
        return (np.cumsum(metric)[running_window:] - np.cumsum(metric)[:-running_window]) / running_window

    @staticmethod
    def get_episode_rewards(dones, rewards):
        df = pd.DataFrame(data={"episode": dones.cumsum(), "reward": rewards})
        df["episode"] = df["episode"].shift().fillna(0)
        episodes_df = df.groupby("episode").agg("sum")
        return episodes_df['reward'].to_numpy()
