import os.path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from hunting_env import HuntingContinuous, EnvParams
from jaxagents.ippo import IPPO, AgentConfig, HyperParameters, OptimizerParams, TrainState, ObsType, ActionType
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import List, Tuple
from agents import PGActorContinuous, PGCritic
from jax_tqdm import scan_tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import pickle


class HuntingIPPO(IPPO):
    log_std = -0.0

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, actor_training: TrainState, state: ObsType)-> Float[Array, "n_actors"]:
        mus = actor_training.apply_fn(actor_training.params, state).squeeze()
        pis = distrax.Normal(loc=mus, scale=jnp.exp(self.log_std))
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0, 4,))
    def _log_prob(self, actor_training: TrainState, params: dict, state: ObsType, actions: ActionType) -> Float[Array, "n_actors"]:
        mus = actor_training.apply_fn(params, state).squeeze()
        log_probs = distrax.Normal(loc=mus, scale=jnp.exp(self.log_std)).log_prob(actions)
        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, actor_training: TrainState, state: ObsType) -> ActionType:
        mus = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        return mus

    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, rng: PRNGKeyArray, actor_training: TrainState, state: ObsType) -> ActionType:
        mus = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        # Use fixed std, OpenAI: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L84
        actions = distrax.Normal(loc=mus, scale=jnp.exp(self.log_std)).sample(seed=rng)
        return actions


def best_predator_win(x):
    pred_win_ratio = jnp.mean(jnp.equal(jnp.take(x["final_rewards"], 1, axis=-1), 10), axis=-1)
    return jnp.argmax(pred_win_ratio).item()


def plot_training(training_metrics, eval_frequency, env_params, path):
    rewards_prey = training_metrics["final_rewards"][..., 0]
    rewards_pred = training_metrics["final_rewards"][..., 1]
    p_prey = np.mean(rewards_prey > env_params.predator_radius, axis=1) * 100
    p_pred = np.mean(-rewards_pred <= env_params.predator_radius, axis=1) * 100
    steps = jnp.arange(rewards_pred.shape[0]) * eval_frequency
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12, 8))
    axs[0, 0].plot(steps, rewards_prey.mean(axis=1), c="b")
    axs[0, 0].fill_between(steps, rewards_prey.min(axis=1), rewards_prey.max(axis=1), color="b", alpha=0.3)
    axs[1, 0].plot(steps, rewards_pred.mean(axis=1), c="r")
    axs[1, 0].fill_between(steps, rewards_pred.min(axis=1), rewards_pred.max(axis=1), color="r", alpha=0.3)
    axs[1, 0].set_xlabel("Training steps", fontsize=12)
    axs[0, 0].set_ylabel("Prey\nFinal reward [-]", fontsize=12)
    axs[1, 0].set_ylabel("Predator\nFinal reward [-]", fontsize=12)
    axs[0, 1].plot(steps, p_prey, c="b")
    axs[1, 1].plot(steps, p_pred, c="r")
    axs[1, 1].set_xlabel("Training steps", fontsize=12)
    axs[0, 1].set_ylabel("Prey\nStalemate ratio [%]", fontsize=12)
    axs[1, 1].set_ylabel("Predator\nWin ratio [%]", fontsize=12)
    axs[0, 1].set_ylim(0, 100)
    axs[1, 1].set_ylim(0, 100)
    for ax in axs.flatten():
        ax.grid()
    plt.close()
    fig.savefig(path)


def plot_loss(training_metrics, eval_frequency, env_params, path):
    actor_loss = training_metrics["actor_loss"].squeeze()
    critic_loss = training_metrics["critic_loss"].squeeze()
    steps = jnp.arange(actor_loss.size) * eval_frequency
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axs[0].plot(steps, actor_loss, c="b")
    axs[0].set_ylabel("Actor loss [-]", fontsize=12)
    axs[1].plot(steps, critic_loss, c="b")
    axs[1].set_xlabel("Training steps", fontsize=12)
    axs[1].set_ylabel("Critic loss [-]", fontsize=12)
    for ax in axs.flatten():
        ax.grid()
    plt.close()
    fig.savefig(path)


def export_csv(render_metrics, csv_path):
    df = pd.DataFrame(
        data = np.round(np.c_[
            render_metrics["step"],
            render_metrics["time"],
            render_metrics["positions"].reshape(n_eval_steps, 4),
            render_metrics["actions"],
            render_metrics["values"],
            render_metrics["next_positions"].reshape(n_eval_steps, 4),
            render_metrics["reward"],
            render_metrics["terminated"],
        ], 2),
    columns=["Step", "Time"]+["Positions_"+str(i+1) for i in range(4)]+["Action1", "Action2", "Value1", "Value2"]+
            ["Next_positions_" + str(i + 1) for i in range(4)]+["Reward1", "Reward2", "Terminated"]
    )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    env_params = EnvParams(prey_velocity=3., predator_velocity=1., max_time=2., caught_reward=100.)
    env = HuntingContinuous()

    folder = r"mlp/prey{prey_vel:.1f}_pred{pred_vel:.1f}_maxtime{max_time:.1f}".format(
        prey_vel=env_params.prey_velocity, pred_vel=env_params.predator_velocity, max_time=env_params.max_time
    )
    if not os.path.exists(folder): os.makedirs(folder)

    fig_folder = os.path.join(folder, "figures")
    if not os.path.exists(fig_folder): os.mkdir(fig_folder)

    if sys.platform == "win32":
        checkpoint_dir = os.path.join("/benchmarks/marl/hunting_2_players", folder, "checkpoints")
    else:
        checkpoint_dir = os.path.join("/mnt/c/Users/mavritsa/Repositories/jax-agents/benchmarks/marl/hunting_2_players", folder, "checkpoints")

    config = AgentConfig(
        n_steps=5_000,
        batch_size=256,
        minibatch_size=16,
        rollout_length=int(env_params.max_time//env_params.dt+1),
        actor_epochs=10,
        critic_epochs=10,
        actor_network=PGActorContinuous,
        critic_network=PGCritic,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
        n_evals=100,
        # checkpoint_dir=checkpoint_dir,
        # restore_agent=False,
    )

    hyperparams = HyperParameters(
        gamma=0.99,
        eps_clip=0.20,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=.005,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=3e-4, eps=1e-5, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=5e-4, eps=1e-5, grad_clip=1)
    )

    ippo = HuntingIPPO(env, env_params, config, eval_during_training=True)
    ippo.log_hyperparams(hyperparams)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    if ippo.eval_during_training:
        ippo.collect_training(runner, training_metrics)
        eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, n_evals=16))

    training_plot_path = os.path.join(fig_folder, "ippo_policy_training_{steps}steps.png".format(steps=config.n_steps))
    plot_training(training_metrics, config.eval_frequency, env_params, training_plot_path)

    training_plot_path = os.path.join(fig_folder, "ippo_losses_{steps}steps.png".format(steps=config.n_steps))
    plot_loss(training_metrics, config.eval_frequency, env_params, training_plot_path)

    def f(runner, i):
        rng, actor_training, critic_training, state, state_env = runner
        actions = ippo.policy(actor_training, state)
        values = critic_training.apply_fn(critic_training.params, state)
        rng, rng_step = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        runner = rng, actor_training, critic_training, next_state, next_env_state
        metrics = {
            "step": i,
            "time": state_env.time,
            "positions": state_env.positions,
            "actions": actions,
            "next_positions": next_env_state.positions,
            "reward": reward,
            "terminated": terminated,
            "values": values,
        }
        return runner, metrics

    n_eval_steps = 500
    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    render_runner = rng, runner.actor_training, runner.critic_training, state, state_env
    render_runner, render_metrics = jax.lax.scan(scan_tqdm(n_eval_steps)(f), render_runner, jnp.arange(n_eval_steps))
    render_metrics = {key: np.asarray(val) for (key, val) in render_metrics.items()}

    csv_path = os.path.join(fig_folder, "details.csv")
    export_csv(render_metrics, csv_path)
    print("Exported csv")

    gif_path = os.path.join(fig_folder, "ippo_{steps}steps.gif".format(steps=config.n_steps))
    env.animate(render_metrics["time"].squeeze(), render_metrics["positions"], render_metrics["actions"],
                render_metrics["values"], env_params, gif_path, export_pdf=True)

    with open(r"../../../../hunting-game/training/models/pretrained/actor_{n_steps}.pkl".format(n_steps=config.n_steps), 'wb') as handle:
        pickle.dump(runner.actor_training.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

