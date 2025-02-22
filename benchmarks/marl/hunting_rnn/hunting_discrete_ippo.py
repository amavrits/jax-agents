import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from hunting_env import HuntingDiscrete, EnvParams
from benchmarks.marl.hunting_rnn.ippo_rnn import IPPO, IPPOConfig, HyperParameters, OptimizerParams, TrainState, STATE_TYPE
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import List, Tuple
from agent_gallery import PGActorDiscrete, PGCritic
from jax_tqdm import scan_tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd


class HuntingIPPO(IPPO):

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, actor_training: TrainState, state: STATE_TYPE, h)-> Float[Array, "n_actors"]:
        logits, _ = actor_training.apply_fn(actor_training.params, state, h)
        pis = distrax.Categorical(logits)
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0,))
    def _log_prob(self, actor_training: TrainState, params: dict, state: STATE_TYPE, actions: Int[Array, "n_actors"], h)\
            -> Float[Array, "n_actors"]:

        logits, _ = actor_training.apply_fn(actor_training.params, state, h)
        actions_onehot = jax.nn.one_hot(actions, 4)
        log_probs = jnp.sum(actions_onehot*logits, axis=1)

        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, actor_training: TrainState, state: STATE_TYPE, h) -> Float[Array, "n_actors"]:
        logits, h = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state, h)
        actions = jnp.argmax(logits, axis=1)
        return actions, h

    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, rng: PRNGKeyArray, actor_training: TrainState, state: STATE_TYPE, h)\
        -> Tuple[PRNGKeyArray, List[Int[Array, "1"]]]:
        """
        Select action by sampling from the stochastic policy for a state.
        :param rng: Random key for initialization.
        :param pi: The distax distribution procuded by the actor network indicating the stochastic policy for a state.
        :return: A random key after action selection and the selected action from the stochastic policy.
        """

        rng_actors = jax.random.split(rng, self.n_actors)

        logits, h = actor_training.apply_fn(actor_training.params, state, h)

        actions = jnp.stack((
            distrax.Categorical(logits=jnp.take(logits, 0, axis=0)).sample(seed=rng_actors[0]),
            distrax.Categorical(logits=jnp.take(logits, 1, axis=0)).sample(seed=rng_actors[1])
        ))

        return actions, h


def best_predator_win(x):
    # terminated = x["terminated"]
    # terminated_pred_rewards = jnp.where(terminated, jnp.take(x["final_rewards"], 1, axis=-1), 0)
    # terminated_pred_wins = jnp.where(jnp.equal(terminated_pred_rewards, 10), 1, 0)
    # pred_win_ratio = jnp.where(
    #     jnp.equal(terminated.sum(axis=-1), 0),
    #     terminated_pred_wins.sum(axis=-1) / terminated.sum(axis=-1),
    #     0
    # )
    pred_win_ratio = jnp.mean(jnp.equal(jnp.take(x["final_rewards"], 1, axis=-1), 10), axis=-1)
    return jnp.argmax(pred_win_ratio).item()


def plot_training(training_metrics, eval_frequency, env_params, path):
    rewards_prey = training_metrics["final_rewards"][..., 0]
    rewards_pred = training_metrics["final_rewards"][..., 1]
    # p_prey = np.mean(rewards_prey != -env_params.caught_reward, axis=1) * 100
    # p_pred = np.mean(rewards_pred == env_params.caught_reward, axis=1) * 100
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

    env_params = EnvParams(prey_velocity=1, predator_velocity=1)
    env = HuntingDiscrete()

    config = IPPOConfig(
        n_steps=1_000,
        seq_length=10,
        hidden_size=32,
        batch_size=64,
        minibatch_size=16,
        rollout_length=100,
        actor_epochs=10,
        critic_epochs=30,
        actor_network=PGActorDiscrete,
        critic_network=PGCritic,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
        n_evals=100,
        # checkpoint_dir=checkpoint_dir,
        # restore_agent=False,
        # restore_agent=True,
    )

    hyperparams = HyperParameters(
        gamma=0.99,
        eps_clip=0.20,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=0.005,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=3e-4, eps=1e-5, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=5e-5, eps=1e-5, grad_clip=1)
    )

    ippo = HuntingIPPO(env, env_params, config, eval_during_training=True)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    # with jax.disable_jit(True): runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, runner.actor_training, n_evals=16))

    training_plot_path = r"figures/discrete/ippo_discrete_policy_training_{steps}steps.png".format(steps=config.n_steps)
    plot_training(training_metrics, config.eval_frequency, env_params, training_plot_path)

    def f(runner, i):
        rng, actor_training, critic_training, state, state_env, actor_hidden_state, critic_hidden_state = runner
        state = jnp.expand_dims(state, axis=0)
        actions, actor_hidden_state = ippo.policy(actor_training, state, actor_hidden_state)
        value, critic_hidden_state = critic_training.apply_fn(critic_training.params, state, critic_hidden_state)
        rng, rng_step = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        next_state = jnp.vstack((state.squeeze(), next_state))
        next_state = jnp.take(next_state, jnp.arange(1, config.seq_length+1), axis=0)
        runner = rng, actor_training, critic_training, next_state, next_env_state, actor_hidden_state, critic_hidden_state
        metrics = {
            "step": i,
            "time": state_env.time.squeeze(),
            "positions": state_env.positions.reshape(-1, 2, 2),
            "actions": actions,
            "next_positions": next_env_state.positions.reshape(-1, 2, 2),
            "reward": reward,
            "values": value,
            "terminated": terminated,
        }
        return runner, metrics

    n_eval_steps = 300
    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    actor_hidden_state = jnp.zeros(config.hidden_size)
    critic_hidden_state = jnp.zeros(config.hidden_size)
    state = jnp.repeat(state, config.seq_length, axis=0)
    render_runner = rng, runner.actor_training, runner.critic_training, state, state_env, actor_hidden_state, critic_hidden_state
    render_runner, render_metrics = jax.lax.scan(scan_tqdm(n_eval_steps)(f), render_runner, jnp.arange(n_eval_steps))
    render_metrics = {key: np.asarray(val) for (key, val) in render_metrics.items()}

    csv_path = r"figures/discrete/details.csv"
    export_csv(render_metrics, csv_path)
    print("Exported csv")

    gif_path = r"figures/discrete/ippo_discrete_policy_{steps}steps.gif".format(steps=config.n_steps)
    env.animate(render_metrics["time"].squeeze(), render_metrics["positions"], render_metrics["actions"],
                render_metrics["values"], env_params, gif_path, export_pdf=True)

