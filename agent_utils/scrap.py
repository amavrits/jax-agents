import numpy as np
import pandas as pd

def _collect_output(self):
    """
    Extracts metrics collected during training and calculates their statistics.
    :return:
    """
    self.dones = np.asarray(self.training_metrics["done"])
    self.rewards = np.asarray(self.training_metrics["reward"])
    self.reward_stats = self._stats(self.rewards)

def get_running_metric(metric, running_window):
    return (np.cumsum(metric)[running_window:] - np.cumsum(metric)[:-running_window]) / running_window

def get_episode_rewards(dones, rewards):
    df = pd.DataFrame(data={"episode": dones.cumsum(), "reward": rewards})
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")
    return episodes_df['reward'].to_numpy()