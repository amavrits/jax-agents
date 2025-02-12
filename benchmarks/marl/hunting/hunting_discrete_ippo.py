import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from hunting_env import HuntingDiscrete, EnvParams
from jaxagents.ippo import IPPO, IPPOConfig, HyperParameters, OptimizerParams, TrainState, STATE_TYPE
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import List, Tuple
from hunting_gallery import PGActorDiscrete, PGCritic
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
    env = HuntingDiscrete()

    config = IPPOConfig(
        n_steps=1_000,
        batch_size=256,
        minibatch_size=16,
        rollout_length=102,
        actor_epochs=50,
        critic_epochs=50,
        actor_network=PGActorDiscrete,
        critic_network=PGCritic,
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
    # eval_metrics = jax.block_until_ready(ippo.eval(rng_eval, runner.actor_training, n_evals=16))

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
            "state": state_env.positions.reshape(-1, 2, 2),
            "actions": actions,
            "next_state": next_env_state.positions.reshape(-1, 2, 2),
            "reward": reward,
            "terminated": terminated,
        }
        return runner, metrics

    n_eval_steps = 300
    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset(rng, env_params)
    eval_runner = rng, runner.actor_training, state, state_env
    eval_runner, metrics = jax.lax.scan(scan_tqdm(n_eval_steps)(f), eval_runner, jnp.arange(n_eval_steps))
    metrics = {key: np.asarray(val) for (key, val) in metrics.items()}

    # training_plot_path = r"figures/ippo_discrete_policy_training_{steps}.png".format(steps=config.n_steps)
    # plot_training(training_metrics, training_plot_path)

    gif_path = r"figures/ippo_discrete_policy_{steps}steps.gif".format(steps=config.n_steps)
    env.animate(metrics["state"], [None, None], env_params, gif_path)

