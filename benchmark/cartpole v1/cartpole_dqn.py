import time
import jax
import optax
from agents import dqn
import gymnax
from cartpole_nn_gallery import *
from postprocessing import PostProcessor


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    transition_temp = dqn.Transition(
        state=jnp.zeros((1, 4), dtype=jnp.float32),
        action=jnp.zeros(1, dtype=jnp.int32),
        reward=jnp.zeros(1, dtype=jnp.float32),
        next_state=jnp.zeros((1, 4), dtype=jnp.float32),
        terminated=jnp.zeros(1, dtype=jnp.bool_),
        info={
            "discount": jnp.array((), dtype=jnp.float32),
            "returned_episode": jnp.array((), dtype=jnp.bool_),
            "returned_episode_lengths": jnp.array((), dtype=jnp.int32),
            "returned_episode_returns": jnp.array((), dtype=jnp.float32),
        }
    )

    def optimizer_fn(optimizer_params):
        return optax.chain(
            optax.clip_by_global_norm(optimizer_params.grad_clip),
            optax.rmsprop(learning_rate=optimizer_params.lr, eps=optimizer_params.eps)
            )

    config = dqn.AgentConfig(
        q_network=DQN_NN_model,
        transition_template=transition_temp,
        n_steps=500,
        buffer_type="FLAT",
        buffer_size=10_000,
        batch_size=128,
        target_update_method="PERIODIC",
        store_agent=False,
        act_randomly=lambda rng, state, n_actions: jax.random.choice(rng, jnp.arange(n_actions)),
        get_performance=lambda i_step, runner: 0,
        set_optimizer=optimizer_fn,
        loss_fn=optax.l2_loss,
        epsilon_type="DECAY",
        epsilon_params=(0.9, 0.05, 50_000)
    )

    # from jax.config import config as jconfig
    # jconfig.update('jax_disable_jax.jit', True)

    agent = dqn.DDQN_Agent(env, env_params, config)

    rng = jax.random.PRNGKey(42)
    t0 = time.time()
    hyperparams = dqn.HyperParameters(0.99, 4, dqn.OptimizerParams(5e-5, 0.01 / 32, 1))
    runner, metrics = agent.train(rng, hyperparams)

    print(f"time: {time.time() - t0:.2f} s")

    pp = PostProcessor(out)
    fig = pp._plot_rewards(N=100)


