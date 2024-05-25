import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import colors
# import matplotlib
# matplotlib.use("TkAgg")


class PostProcessor:

    def __init__(self, results):
        self.last_runner = results["runner"]
        self.metrics = results["metrics"]
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

    def _plot_rewards(self, N=2_000):
        episode_rewards = self._episode_rewards(self.dones, self.step_rewards)
        running_rewards = (np.cumsum(episode_rewards)[N:] - np.cumsum(episode_rewards)[:-N]) / N

        fig = plt.figure()
        plt.plot(episode_rewards, c='b', alpha=0.4)
        plt.plot(np.arange(N, episode_rewards.size), running_rewards, c='b')
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Reward [-]", fontsize=14)

        return fig

    @staticmethod
    def _viz_policy(policy):

        policy = policy.reshape(-1, 10)

        fig, ax = plt.subplots(figsize=(12, 8))

        cmap = colors.ListedColormap(['royalblue', 'limegreen', 'firebrick', 'orange'])
        bounds = [0, 0.5, 1.5, 2.5, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(policy, cmap=cmap, norm=norm)
        ax.set_xlabel('Dealer hand', fontsize=12)
        ax.set_ylabel('Player hand', fontsize=12, labelpad=12)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        ticks_loc = [i - 2 for i in range(2, policy.shape[1] + 2)]
        ax.set_xticks(ticks_loc)
        ax.set_xticklabels(list(policy.columns))

        ticks_loc = [i - 4 for i in range(4, policy.shape[0] + 4)]
        ax.set_yticks(ticks_loc)
        ax.set_yticklabels(list(policy.index))

        ax.set_xticks(np.arange(-.5, len(policy.columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(policy.index), 1), minor=True)
        ax.grid(which='minor', color='k', linewidth=.5)

        cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Stand', 'Hit', 'Double', 'Split'])
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Best action', rotation=270)
        plt.close()

        return fig


