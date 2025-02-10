import jax
import jax.numpy as jnp
import optax
from hunting_env import HuntingDiscrete, EnvParams
from jaxagents import ippo
from hunting_nn_gallery import PGActorNNDiscrete, PGCriticNN
import matplotlib.pyplot as plt


if __name__ == "__main__":

    env_params = EnvParams(prey_velocity=2, predator_velocity=1)
    env = HuntingDiscrete()

    agents = (ippo.IPPOAgent(PGActorNNDiscrete, PGCriticNN), ippo.IPPOAgent(PGActorNNDiscrete, PGCriticNN))

    config = ippo.IPPOConfig(
        n_steps=100,
        batch_size=16,
        minibatch_size=4,
        rollout_length=100,
        actor_epochs=10,
        critic_epochs=10,
        optimizers=[optax.adam]*len(agents),
        eval_frequency=10,
        eval_rng=jax.random.PRNGKey(18),
    )

    hyperparams = ippo.HyperParameters(
        gamma=0.99,
        eps_clip=0.2,
        kl_threshold=1e-5,
        gae_lambda=0.97,
        ent_coeff=0.0,
        vf_coeff=1.0,
        actor_optimizer_params=ippo.OptimizerParams(learning_rate=3e-4, eps=1e-3, grad_clip=1),
        critic_optimizer_params=ippo.OptimizerParams(learning_rate=1e-3, eps=1e-3, grad_clip=1)
    )

    ippo = ippo.IPPO(env, env_params, agents, config, eval_during_training=False)

    rng = jax.random.PRNGKey(42)
    rng_train, rng_eval = jax.random.split(rng)
    runner, training_metrics = jax.block_until_ready(ippo.train(rng_train, hyperparams))
    # eval_metrics = ippo.eval(rng_eval, runner.actor_trainings, n_evals=16)
    eval_metrics = ippo._eval_agent(rng_eval, runner.actor_trainings, 16)

    # returns_prey = training_metrics["episode_returns"][0]
    # returns_pred = training_metrics["episode_returns"][1]
    # steps = jnp.arange(1, returns_pred.size+1)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax.plot(steps, returns_prey.mean(1), c="b")
    # ax.fill_between(steps, returns_prey.min(1), returns_prey.max(1), color="b")
    # ax2.plot(steps, returns_pred.mean(1), c="r")
    # ax2.fill_between(steps, returns_pred.min(1), returns_pred.max(1), color="r")
    # ax.set_xlabel("Training steps", fontsize=12)
    # ax.set_ylabel("Prey returns", fontsize=12)
    # ax2.set_ylabel("Predator returns", fontsize=12)
    # plt.close()
    # fig.savefig(r"figures/ippo_training.png")

    rng = jax.random.PRNGKey(43)
    state, state_env = env.reset_env(rng, env_params)

    figs = []
    fig = env.render(state, jnp.zeros(2), env_params)
    figs.append(fig)

    done = False
    rewards = []
    step = 0
    while not done:
        step += 1
        actions = ippo.policy(runner.actor_trainings, state)
        rng, rng_step = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info = env.step(rng_step, state_env, actions, env_params)
        done = terminated or step > 250
        state = next_state
        state_env = next_env_state
        rewards.append(reward)
        fig = env.render(next_state, actions, env_params)
        figs.append(fig)

    import numpy as np
    rewards = np.asarray(rewards)

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(r"figures/ippo_discrete_render.pdf")
    [pp.savefig(fig) for fig in figs]
    pp.close()


    import matplotlib.animation as animation
    from PIL import Image
    import io

    image_frames = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        image_frames.append(img)
        plt.close(fig)

    gif_path = r"figures/ippo_discrete_policy_{steps}.gif".format(steps=config.n_steps)
    image_frames[0].save(
        gif_path,
        save_all=True,
        append_images=image_frames[1:],
        duration=50,  # Duration for each frame (in milliseconds)
        loop=0  # Loop 0 means infinite loop
    )


