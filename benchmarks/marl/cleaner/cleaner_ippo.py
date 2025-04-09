import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from cleaner_env import Cleaner
from jaxagents.ippo import IPPO, IPPOConfig, HyperParameters, OptimizerParams, TrainState, STATE_TYPE
from jaxtyping import Array, Float, Int, PRNGKeyArray, Bool
from typing import List, Tuple
from agent_gallery import PGActorDiscrete, PGCritic
from jax_tqdm import scan_tqdm
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd


class CleanerIPPO(IPPO):

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, actor_training: TrainState, state: STATE_TYPE)-> Float[Array, "n_actors"]:
        logits = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        pis = distrax.Categorical(logits)
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0, 4,))
    def _log_prob(self, actor_training: TrainState, params: dict, state: STATE_TYPE, actions: Int[Array, "n_actors"])\
            -> Float[Array, "n_actors"]:
        logits = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        log_probs = distrax.Categorical(logits).log_prob(actions)
        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, actor_training: TrainState, state: STATE_TYPE) -> Float[Array, "n_actors"]:
        logits = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        return jnp.argmax(logits, axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, rng: PRNGKeyArray, actor_training: TrainState, state: STATE_TYPE)\
        -> Tuple[PRNGKeyArray, List[Int[Array, "1"]]]:
        logits = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state).squeeze()
        actions = distrax.Categorical(logits).sample(seed=rng)
        return actions


def plot_training(training_metrics, eval_frequency, env_params, path):
    rewards = training_metrics["sum_rewards"][..., 0]
    steps = jnp.arange(rewards.shape[0]) * eval_frequency
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(steps, rewards.mean(axis=1), c="b")
    ax.fill_between(steps, rewards.min(axis=1), rewards.max(axis=1), color="b", alpha=0.3)
    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("Accumulated reward [-]", fontsize=12)
    ax.grid()
    plt.close()
    fig.savefig(path)


def export_csv(render_metrics, csv_path):
    df = pd.DataFrame(
        data = np.round(np.c_[
            render_metrics["step"],
            # render_metrics["state"].reshape(n_eval_steps, -1),
            render_metrics["actions"],
            render_metrics["values"],
            # render_metrics["next_state"].reshape(n_eval_steps, -1),
            render_metrics["reward"][..., 0],
            render_metrics["terminated"],
        ], 2),
    columns=["Step", "Action1", "Action2", "Action3", "Value1", "Value2", "Value3", "Reward", "Terminated"]
    # columns=["Step", "Time"]+["Positions_"+str(i+1) for i in range(4)]+["Action1", "Action2", "Value1", "Value2"]+
    #         ["Next_positions_" + str(i + 1) for i in range(4)]+["Reward1", "Reward2", "Terminated"]
    )
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":

    env_params = ()
    env = Cleaner()

    if sys.platform == "win32":
        checkpoint_dir = "C:\\Users\\mavritsa\\Repositories\\jax-agents\\benchmarks\\marl\\hunting_mlp\\checkpoints\\ippo\\continuous"
    else:
        checkpoint_dir = "/mnt/c/Users/mavritsa/Repositories/jax-agents/benchmarks/marl/hunting_mlp/checkpoints/ippo/continuous"

    config = IPPOConfig(
        n_steps=1_000,
        batch_size=32,
        minibatch_size=4,
        rollout_length=100,
        actor_epochs=10,
        critic_epochs=10,
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
        ent_coeff=0.5,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=3e-4, eps=1e-5, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=5e-5, eps=1e-5, grad_clip=1)
    )

    ippo = CleanerIPPO(env, env_params, config, eval_during_training=True)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, runner.actor_training, n_evals=16))

    training_plot_path = r"figures/ippo_continuous_policy_training_{steps}steps.png".format(steps=config.n_steps)
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
            "state": state,
            "actions": actions,
            "next_state": next_state,
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

    csv_path = r"figures/details.csv"
    export_csv(render_metrics, csv_path)
    print("Exported csv")

    # gif_path = r"figures/ippo_continuous_policy_{steps}steps.gif".format(steps=config.n_steps)
    # env.animate(render_metrics["time"].squeeze(), render_metrics["positions"], render_metrics["actions"],
    #             render_metrics["values"], env_params, gif_path, export_pdf=True)

