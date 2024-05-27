import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
import matplotlib
matplotlib.use("TkAgg")


class PostProcessor:

    def __init__(self, runner, metrics):
        self.last_runner = runner
        self.metrics = metrics
        self._collect_output()

    def _collect_output(self):
        self.dones = np.asarray(self.metrics["done"])
        self.step_rewards = np.asarray(self.metrics["reward"])

    @staticmethod
    def _episode_rewards(dones, step_rewards):
        df = pd.DataFrame(data={"episode": dones.cumsum(), "reward": step_rewards})
        df["episode"] = df["episode"].shift().fillna(0)
        episodes_df = df.groupby("episode").agg("sum")
        episode_rewards = episodes_df['reward'].to_numpy()
        return episode_rewards

    @staticmethod
    def _policy(q_network, params, mesh):
        Q_table = q_network.apply(params, mesh)
        Q_table = np.asarray(Q_table)
        policy = np.argmax(Q_table, axis=1)
        return policy

    def _plot_rewards(self, running_window=2_000, close_plot=False):
        episode_rewards = self._episode_rewards(self.dones, self.step_rewards)
        running_rewards = (np.cumsum(episode_rewards)[running_window:] - np.cumsum(episode_rewards)[:-running_window]) / running_window

        fig = plt.figure()
        plt.plot(episode_rewards, c='b', alpha=0.4)
        plt.plot(np.arange(running_window, episode_rewards.size), running_rewards, c='b')
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward [-]", fontsize=14)

        if close_plot: plt.close()

        return fig
