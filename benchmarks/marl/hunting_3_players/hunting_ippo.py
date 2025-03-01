import os.path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from hunting3_env import HuntingContinuous, EnvParams
from jaxagents.ippo import IPPO, IPPOConfig, HyperParameters, OptimizerParams, TrainState, STATE_TYPE
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import List, Tuple
from agents import PGActorContinuous, PGCritic
from jax_tqdm import scan_tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd


class HuntingIPPO(IPPO):
    log_std = -0.0

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, actor_training: TrainState, state: STATE_TYPE)-> Float[Array, "n_actors"]:
        mus = actor_training.apply_fn(actor_training.params, state).squeeze()
        pis = distrax.Normal(loc=mus, scale=jnp.exp(self.log_std))
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0, 4,))
    def _log_prob(self, actor_training: TrainState, params: dict, state: STATE_TYPE, actions: Int[Array, "n_actors"])\
            -> Float[Array, "n_actors"]:
        mus = actor_training.apply_fn(params, state).squeeze()
        log_probs = distrax.Normal(loc=mus, scale=jnp.exp(self.log_std)).log_prob(actions)
        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, actor_training: TrainState, state: STATE_TYPE) -> Float[Array, "n_actors"]:
        mus = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        return mus

    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, rng: PRNGKeyArray, actor_training: TrainState, state: STATE_TYPE)\
        -> Tuple[PRNGKeyArray, List[Int[Array, "1"]]]:
        mus = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        # Use fixed std, OpenAI: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L84
        actions = distrax.Normal(loc=mus, scale=jnp.exp(self.log_std)).sample(seed=rng)
        return actions


def best_predator_win(x):
    pred_win_ratio = jnp.mean(jnp.equal(jnp.take(x["final_rewards"], 1, axis=-1), 10), axis=-1)
    return jnp.argmax(pred_win_ratio).item()


def plot_training(training_metrics, eval_frequency, env_params, path):
    rewards_prey = training_metrics["final_rewards"][..., 0]
    rewards_pred1 = training_metrics["final_rewards"][..., 1]
    rewards_pred2 = training_metrics["final_rewards"][..., 2]
    p_prey = np.mean(rewards_prey > max(env_params.predator_radius), axis=1) * 100
    p_pred1 = np.mean(-rewards_pred1 <= env_params.predator_radius[0], axis=1) * 100
    p_pred2 = np.mean(-rewards_pred2 <= env_params.predator_radius[1], axis=1) * 100
    steps = jnp.arange(rewards_pred1.shape[0]) * eval_frequency
    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    axs[0, 0].plot(steps, rewards_prey.mean(axis=1), c="b")
    axs[1, 0].plot(steps, rewards_pred1.mean(axis=1), c="r")
    axs[2, 0].plot(steps, rewards_pred1.mean(axis=1), c="g")
    axs[0, 0].fill_between(steps, rewards_prey.min(axis=1), rewards_prey.max(axis=1), color="b", alpha=0.3)
    axs[1, 0].fill_between(steps, rewards_pred1.min(axis=1), rewards_pred1.max(axis=1), color="r", alpha=0.3)
    axs[2, 0].fill_between(steps, rewards_pred2.min(axis=1), rewards_pred2.max(axis=1), color="g", alpha=0.3)
    axs[2, 0].set_xlabel("Training steps", fontsize=12)
    axs[0, 0].set_ylabel("Prey\nFinal reward [-]", fontsize=12)
    axs[1, 0].set_ylabel("Predator 1\nFinal reward [-]", fontsize=12)
    axs[2, 0].set_ylabel("Predator 2\nFinal reward [-]", fontsize=12)
    axs[0, 1].plot(steps, p_prey, c="b")
    axs[1, 1].plot(steps, p_pred1, c="r")
    axs[2, 1].plot(steps, p_pred2, c="g")
    axs[2, 1].set_xlabel("Training steps", fontsize=12)
    axs[0, 1].set_ylabel("Prey\nStalemate ratio [%]", fontsize=12)
    axs[1, 1].set_ylabel("Predator 1\nWin ratio [%]", fontsize=12)
    axs[2, 1].set_ylabel("Predator 2\nWin ratio [%]", fontsize=12)
    axs[0, 1].set_ylim(0, 100)
    axs[1, 1].set_ylim(0, 100)
    axs[2, 1].set_ylim(0, 100)
    for ax in axs.flatten():
        ax.grid()
    plt.close()
    fig.savefig(path)


def export_csv(render_metrics, csv_path):
    df = pd.DataFrame(
        data = np.round(np.c_[
            render_metrics["step"],
            render_metrics["time"],
            render_metrics["positions"].reshape(n_eval_steps, 6),
            render_metrics["actions"],
            render_metrics["values"],
            render_metrics["next_positions"].reshape(n_eval_steps, 6),
            render_metrics["reward"],
            render_metrics["terminated"],
        ], 2),
    columns=["Step", "Time"]+["Positions_"+str(i+1) for i in range(6)]+["Action1", "Action2", "Action3", "Value1", "Value2", "Value3"]+
            ["Next_positions_" + str(i + 1) for i in range(6)]+["Reward1", "Reward2", "Reward3", "Terminated"]
    )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    env_params = EnvParams(
        prey_velocity=3.,
        predator_velocity=(1., 1.),
        predator_radius=(.1, .1),
        max_time=2.
    )
    env = HuntingContinuous()

    folder = r"mlp/prey{prey_vel:.1f}_pred1_{pred_vel1:.1f}_pred2_{pred_vel2:.1f}_maxtime{max_time:.1f}".format(
        prey_vel=env_params.prey_velocity,
        pred_vel1=env_params.predator_velocity[0],
        pred_vel2=env_params.predator_velocity[1],
        max_time=env_params.max_time
    )
    if not os.path.exists(folder): os.makedirs(folder)

    fig_folder = os.path.join(folder, "figures")
    if not os.path.exists(fig_folder): os.mkdir(fig_folder)

    if sys.platform == "win32":
        checkpoint_dir = os.path.join("/benchmarks/marl/hunting_3_players", folder, "checkpoints")
    else:
        checkpoint_dir = os.path.join("/mnt/c/Users/mavritsa/Repositories/jax-agents/benchmarks/marl/hunting_3_players", folder, "checkpoints")

    config = IPPOConfig(
        n_steps=30_000,
        batch_size=256,
        minibatch_size=32,
        rollout_length=int(env_params.max_time//env_params.dt+1),
        actor_epochs=10,
        critic_epochs=10,
        actor_network=PGActorContinuous,
        critic_network=PGCritic,
        optimizer=optax.adam,
        eval_frequency=300,
        eval_rng=jax.random.PRNGKey(18),
        n_evals=500,
        # checkpoint_dir=checkpoint_dir,
        # restore_agent=False,
    )

    hyperparams = HyperParameters(
        gamma=0.99,
        eps_clip=0.2,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=.005,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=3e-4, eps=1e-5, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=5e-5, eps=1e-5, grad_clip=1)
    )

    ippo = HuntingIPPO(env, env_params, config, eval_during_training=True)
    ippo.log_hyperparams(hyperparams)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, runner.actor_training, n_evals=16))

    training_plot_path = os.path.join(fig_folder, "ippo_policy_training_{steps}steps.png".format(steps=config.n_steps))
    plot_training(training_metrics, config.eval_frequency, env_params, training_plot_path)

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

