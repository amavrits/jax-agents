import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from hunting_env import HuntingContinuous, EnvParams
from jaxagents.ippo_NEW import IPPO, IPPOConfig, HyperParameters, OptimizerParams, TrainState, STATE_TYPE
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import List, Tuple
from hunting_nn_gallery import PGActorNNContinuousMA, PGCriticNNMA
from jax_tqdm import scan_tqdm
from functools import partial
import matplotlib.pyplot as plt


class HuntingIPPO(IPPO):

    @partial(jax.jit, static_argnums=(0,))
    def _entropy(self, actor_training: TrainState, state: STATE_TYPE)-> Float[Array, "n_agents"]:
        params = actor_training.apply_fn(actor_training.params, state)
        # pis = distrax.Normal(loc=jnp.take(params, 0, axis=-1), scale=jnp.exp(jnp.take(params, 1, axis=-1)))
        pis = distrax.Normal(loc=jnp.take(params, 0, axis=-1), scale=3.)
        return pis.entropy()

    @partial(jax.jit, static_argnums=(0,))
    def _log_prob(self, actor_training: TrainState, state: STATE_TYPE, actions: Int[Array, "n_agents"])\
            -> Float[Array, "n_agents"]:
        params = actor_training.apply_fn(actor_training.params, state)
        # actions_base = jnp.arctanh(2*actions-1)
        # actions_base = jnp.arctanh(actions)
        # log_probs = distrax.Normal(loc=jnp.take(params, 0, axis=-1), scale=jnp.exp(jnp.take(params, 1, axis=-1))).log_prob(actions_base)
        log_probs = distrax.Normal(loc=jnp.take(params, 0, axis=-1), scale=3.).log_prob(actions)
        return log_probs

    @partial(jax.jit, static_argnums=(0,))
    def _mode(self, actor_training: TrainState, state: STATE_TYPE) -> Float[Array, "n_agents"]:
        params = actor_training.apply_fn(jax.lax.stop_gradient(actor_training.params), state)
        actions_base = jnp.take(params, 0, axis=-1)
        # actions = 0.5 * (1 + jax.nn.tanh(actions_base))
        # actions = jax.nn.tanh(actions_base)
        # actions = (jnp.take(params, 0, axis=-1) - 1) / (jnp.take(params, 0, axis=-1) + jnp.take(params, 1, axis=-1) - 2)
        return actions_base

    @partial(jax.jit, static_argnums=(0,))
    def _sample_actions(self, rng: PRNGKeyArray, actor_training: TrainState, state: STATE_TYPE)\
        -> Tuple[PRNGKeyArray, List[Int[Array, "1"]]]:
        """
        Select action by sampling from the stochastic policy for a state.
        :param rng: Random key for initialization.
        :param pi: The distax distribution procuded by the actor network indicating the stochastic policy for a state.
        :return: A random key after action selection and the selected action from the stochastic policy.
        """

        n_actors = 2

        rng, *rng_actors = jax.random.split(rng, n_actors+1)

        params = actor_training.apply_fn(actor_training.params, state)

        # params = jnp.exp(params)

        # actions_base = jnp.stack((
        #     distrax.Normal(
        #         loc=jnp.take(jnp.take(params, 0, axis=0), 0, axis=-1),
        #         scale=jnp.take(jnp.take(params, 0, axis=0), 1, axis=-1)
        #     ).sample(seed=rng_actors[0]),
        #     distrax.Normal(
        #         loc=jnp.take(jnp.take(params, 1, axis=0), 0, axis=-1),
        #         scale=jnp.take(jnp.take(params, 1, axis=0), 1, axis=-1)
        #     ).sample(seed=rng_actors[1]),
        # ))

        actions_base = distrax.Normal(
            loc=jnp.take(params, 0, axis=-1),
            # scale=jnp.exp(jnp.take(params, 1, axis=-1))
            scale=3.
            ).sample(seed=rng_actors[0])

        # actions = 0.5 * (1 + jax.nn.tanh(actions_base))
        # actions = jax.nn.tanh(actions_base)

        return actions_base


def plot_training(training_metrics, path):
    returns_prey = training_metrics["episode_returns"][0]
    returns_pred = training_metrics["episode_returns"][1]
    steps = jnp.arange(1, returns_pred.size+1)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(steps, returns_prey.mean(1), c="b")
    ax.fill_between(steps, returns_prey.min(1), returns_prey.max(1), color="b")
    ax2.plot(steps, returns_pred.mean(1), c="r")
    ax2.fill_between(steps, returns_pred.min(1), returns_pred.max(1), color="r")
    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("Prey returns", fontsize=12)
    ax2.set_ylabel("Predator returns", fontsize=12)
    plt.close()
    fig.savefig(path)


if __name__ == "__main__":

    env_params = EnvParams(prey_velocity=2, predator_velocity=1)
    env = HuntingContinuous()

    config = IPPOConfig(
        n_steps=1_000,
        batch_size=256,
        minibatch_size=16,
        rollout_length=201,
        actor_epochs=50,
        critic_epochs=50,
        actor_network=PGActorNNContinuousMA,
        critic_network=PGCriticNNMA,
        optimizer=optax.adam,
        eval_frequency=100,
        eval_rng=jax.random.PRNGKey(18),
    )

    hyperparams = HyperParameters(
        gamma=0.99,
        eps_clip=0.05,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=0.001,
        vf_coeff=1.0,
        actor_optimizer_params=OptimizerParams(learning_rate=1e-3, eps=1e-3, grad_clip=1),
        critic_optimizer_params=OptimizerParams(learning_rate=1e-3, eps=1e-3, grad_clip=1)
    )

    ippo = HuntingIPPO(env, env_params, config, eval_during_training=False)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    # with jax.disable_jit(True): runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    # eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, runner.actor_training, n_evals=16))

    def f(runner, i):
        rng, actor_training, state, state_env = runner
        params = actor_training.apply_fn(actor_training.params, state)
        actions = ippo.policy(actor_training, state)
        rng, rng_step = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        runner = rng, actor_training, next_state, next_env_state
        m = {
            "step": i,
            "params": params,
            "state": state_env.positions.reshape(-1, 2, 2),
            "actions": actions,
            "next_state": next_env_state.positions.reshape(-1, 2, 2),
            "reward": reward,
            "terminated": terminated,
        }
        return runner, m

    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    eval_runner = rng, runner.actor_training, state, state_env
    eval_runner, metrics = jax.lax.scan(scan_tqdm(300)(f), eval_runner, jnp.arange(300))
    metrics = {key: np.asarray(val) for (key, val) in metrics.items()}
    actions = metrics["actions"]
    params = metrics["params"].squeeze()

    # training_plot_path = r"figures/ippo_continuous_policy_training_{steps}.png".format(steps=config.n_steps)
    # plot_training(training_metrics, training_plot_path)

    gif_path = r"figures/ippo_continuous_policy_{steps}.gif".format(steps=config.n_steps)
    env.animate(metrics["state"], [None, None], env_params, gif_path)

