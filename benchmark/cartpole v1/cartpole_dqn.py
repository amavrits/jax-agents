import sys
import time
import jax
import optax
import gymnax
import numpy as np
from cartpole_nn_gallery import *

sys.path.append('./')
try:
    from agents import dqn
    from agent_utils.q_agent_eval import QAgentEvaluator
except:
    raise


if __name__ == '__main__':

    env, env_params = gymnax.make("CartPole-v1")

    """Set up transition template, given the state representation in the cartpole environment"""
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

    """Set up function for initializing the optimizer"""
    def optimizer_fn(optimizer_params):
        return optax.chain(
            optax.clip_by_global_norm(optimizer_params.grad_clip),
            optax.rmsprop(learning_rate=optimizer_params.lr, eps=optimizer_params.eps)
            )

    """Define configuration for agent training"""
    config = dqn.AgentConfig(
        q_network=DQN_NN_model,
        transition_template=transition_temp,
        n_steps=50000,
        buffer_type="FLAT",
        buffer_size=10_000,
        batch_size=128,
        target_update_method="PERIODIC",
        store_agent=False,
        act_randomly=lambda rng, state, n_actions: jax.random.choice(rng, jnp.arange(n_actions)),
        get_performance=lambda i_step, runner: 0,
        set_optimizer=optimizer_fn,
        loss_fn=optax.l2_loss,
        epsilon_fn_style="DECAY",
        epsilon_params=(0.9, 0.05, 50_000)
    )

    """Set up agent"""
    agent = dqn.DDQN_Agent(env, env_params, config)
    print(agent.__str__())

    """Define optimizer parameters and training hyperparameters"""
    optimizer_params = dqn.OptimizerParams(5e-5, 0.01 / 32, 1)
    hyperparams = dqn.HyperParameters(0.99, 4, optimizer_params)

    """Draw random key"""
    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)

    """Train agent"""
    t0 = time.time()
    runner, metrics = agent.train(rng_train, hyperparams)
    print(f"time: {time.time() - t0:.2f} s")

    """ Post-process results"""
    agent.collect_train(runner, metrics)

    """Evaluate agent performance"""
    eval_metrics = agent.eval(rng_eval, n_evals=100)

    """ Plot results"""
    running_window = 100
    episode_rewards = get_episode_rewards(agent.training_metrics["done"], agent.training_metrics["reward"])
    running_rewards = get_running_metric(episode_rewards, running_window)


    A = np.asarray(agent.training_metrics["done"])
    AA = np.asarray(agent.training_metrics["reward"])

    df = pd.DataFrame(data={"episode": agent.training_metrics["done"].cumsum(), "reward": agent.training_metrics["reward"]})
    df["episode"] = df["episode"].shift().fillna(0)
    episodes_df = df.groupby("episode").agg("sum")





    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(episode_rewards, c='b', alpha=0.4)
    plt.plot(np.arange(running_window, episode_rewards.size), running_rewards, c='b')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward [-]", fontsize=14)
    plt.close()
    fig.savefig(r'C:\Users\mavritsa\OneDrive - Stichting Deltares\Desktop\AAA.png')

