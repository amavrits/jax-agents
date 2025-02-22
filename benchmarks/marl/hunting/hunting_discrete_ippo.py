import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from hunting_env import HuntingDiscrete, EnvParams
from jaxagents.ippo import IPPO, IPPOConfig, HyperParameters, OptimizerParams, TrainState, STATE_TYPE
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import List, Tuple
from agent_gallery import PGActorDiscrete, PGCritic
from jax_tqdm import scan_tqdm
from functools import partial
import matplotlib.pyplot as plt


class HuntingIPPO(IPPO):

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, actor_training: TrainState, state: STATE_TYPE)-> Float[Array, "n_actors"]:
        logits = actor_training.apply_fn(actor_training.params, state)
        pis = distrax.Categorical(logits)
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0,))
    def _log_prob(self, actor_training: TrainState, state: STATE_TYPE, actions: Int[Array, "n_actors"])\
            -> Float[Array, "n_actors"]:

        logits = actor_training.apply_fn(actor_training.params, state)
        actions_onehot = jax.nn.one_hot(actions, 4)
        log_probs = jnp.sum(actions_onehot*logits, axis=1)

        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, actor_training: TrainState, state: STATE_TYPE) -> Float[Array, "n_actors"]:
        logits = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state)
        actions = jnp.argmax(logits, axis=1)
        return actions

    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, rng: PRNGKeyArray, actor_training: TrainState, state: STATE_TYPE)\
        -> Tuple[PRNGKeyArray, List[Int[Array, "1"]]]:
        """
        Select action by sampling from the stochastic policy for a state.
        :param rng: Random key for initialization.
        :param pi: The distax distribution procuded by the actor network indicating the stochastic policy for a state.
        :return: A random key after action selection and the selected action from the stochastic policy.
        """

        rng_actors = jax.random.split(rng, self.n_actors)

        logits = actor_training.apply_fn(actor_training.params, state)

        actions = jnp.stack((
            distrax.Categorical(logits=jnp.take(logits, 0, axis=0)).sample(seed=rng_actors[0]),
            distrax.Categorical(logits=jnp.take(logits, 1, axis=0)).sample(seed=rng_actors[1])
        ))

        return actions


def plot_training(training_metrics, eval_frequency, env_params, path):
    rewards_prey = training_metrics["final_rewards"][..., 0]
    rewards_pred = training_metrics["final_rewards"][..., 1]
    p_prey = np.mean(training_metrics["final_rewards"][..., 0] != -env_params.caught_reward, axis=1) * 100
    p_pred = np.mean(training_metrics["final_rewards"][..., 1] == env_params.caught_reward, axis=1) * 100
    steps = jnp.arange(1, rewards_pred.shape[0]+1) * eval_frequency
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
    for ax in axs.flatten():
        ax.grid()
    plt.close()
    fig.savefig(path)


if __name__ == "__main__":

    env_params = EnvParams(prey_velocity=2, predator_velocity=1)
    env = HuntingDiscrete()

    config = IPPOConfig(
        n_steps=1_000,
        batch_size=256,
        minibatch_size=16,
        rollout_length=100,
        actor_epochs=50,
        critic_epochs=50,
        actor_network=PGActorDiscrete,
        critic_network=PGCritic,
        optimizer=optax.adam,
        eval_frequency=5,
        eval_rng=jax.random.PRNGKey(18),
    )

    hyperparams = HyperParameters(
        gamma=0.99,
        eps_clip=0.05,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=0.001,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=1e-4, eps=1e-3, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=1e-3, eps=1e-3, grad_clip=1)
    )

    ippo = HuntingIPPO(env, env_params, config, eval_during_training=True)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, runner.actor_training, n_evals=16))

    def f(runner, i):
        rng, actor_training, state, state_env = runner
        params = actor_training.apply_fn(actor_training.params, state)
        actions = ippo.policy(actor_training, state)
        rng, rng_step = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        runner = rng, actor_training, next_state, next_env_state
        metrics = {
            "step": i,
            "time": state_env.time,
            "params": params,
            "positions": state_env.positions.reshape(-1, 2, 2),
            "actions": actions,
            "next_positions": next_env_state.positions.reshape(-1, 2, 2),
            "reward": reward,
            "terminated": terminated,
        }
        return runner, metrics

    n_eval_steps = 300
    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    render_runner = rng, runner.actor_training, state, state_env
    render_runner, render_metrics = jax.lax.scan(scan_tqdm(n_eval_steps)(f), render_runner, jnp.arange(n_eval_steps))
    render_metrics = {key: np.asarray(val) for (key, val) in render_metrics.items()}

    training_plot_path = r"figures/discrete/ippo_discrete_policy_training_{steps}steps.png".format(steps=config.n_steps)
    plot_training(training_metrics, config.eval_frequency, env_params, training_plot_path)

    gif_path = r"figures/discrete/ippo_discrete_policy_{steps}steps.gif".format(steps=config.n_steps)
    env.animate(render_metrics["positions"], render_metrics["actions"], env_params, gif_path)

