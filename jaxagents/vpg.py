"""
Implementation of the Vanilla Policy Gradient agent in JAX.

Author: Antonis Mavritsakis
@Github: amavrits

References
----------
.. [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013).
       Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

"""


import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm
import optax
import distrax
import xarray as xr
from flax.core import FrozenDict
import flashbax as fbx
from jaxagents.agent_utils.vpg_utils import *
from gymnax.environments.environment import Environment, EnvParams
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper, LogEnvState
from flax.training.train_state import TrainState
from abc import abstractmethod
from functools import partial
from abc import ABC
from typing import Tuple, Dict, NamedTuple, Type, Union, Optional, ClassVar
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
import warnings

warnings.filterwarnings("ignore")

BufferStateType = fbx.trajectory_buffer.BufferState


class VPGAgent(ABC):
    """
    Vanilla Policy Gradient agent

    Training relies on jitting several methods by treating the 'self' arg as static. According to suggested practice,
    this can prove dangerous (https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods -
    How to use jit with methods?); if attrs of 'self' change during training, the changes will not be registered in
    jit. In this case, neither agent training nor evaluation change any 'self' attrs, so using Strategy 2 of the
    suggested practice is valid. Otherwise, strategy 3 should have been used.
    """

    agent_trained: ClassVar[bool] = False  # Whether the agent has been trained.
    # Optimal policy network parameters after post-processing.
    agent_params: ClassVar[Optional[Union[Dict, FrozenDict]]] = None
    training_runner: ClassVar[Optional[UpdateRunner]] = None  # Runner object after training.
    training_metrics: ClassVar[Optional[Dict]] = None  # Metrics collected during training.

    def __init__(self, env: Type[Environment], env_params: EnvParams, config: AgentConfig) -> None:
        """
        :param env: A gymnax or custom environment that inherits from the basic gymnax class.
        :param env_params: A dataclass named "EnvParams" containing the parametrization of the environment.
        :param config: The configuration of the agent as one of the following objects: AgentConfig,
                       CategoricalAgentConfig, QuantileAgentConfig. For more information
                       on these objects check dqn_utils. The selected object must match the agent.
        """

        self.config = config
        self._init_env(env, env_params)

    def __str__(self) -> str:
        """
        Returns a string containing only the non-default field values.
        """

        output_lst = [field + ': ' + str(getattr(self.config, field)) for field in self.config._fields]
        output_lst = ['Agent configuration:'] + output_lst

        return '\n'.join(output_lst)


    """ GENERAL METHODS"""

    def _init_env(self, env: Type[Environment], env_params: EnvParams) -> None:
        """
        Environment initialization.
        :param env: A gymnax or custom environment that inherits from the basic gymnax class.
        :param env_params: A dataclass containing the parametrization of the environment.
        :return:
        """

        env = FlattenObservationWrapper(env)
        self.env = LogWrapper(env)
        self.env_params = env_params
        self.n_actions = self.env.action_space(self.env_params).n

    @partial(jax.jit, static_argnums=(0,))
    def _init_buffer(self) -> BufferStateType:
        """
        Buffer initialization.
        :return: The initialized buffer for the agent.
        """

        # if self.config.buffer_type == "FLAT":

        # self.buffer_fn = fbx.make_flat_buffer(
        #     max_length=self.config.buffer_size,
        #     min_length=self.config.batch_size,
        #     sample_batch_size=self.config.batch_size,
        #     add_sequences=True,
        #     add_batch_size=None,
        # )
        # buffer_state = self.buffer_fn.init(self.config.transition_template)

        self.buffer_fn = fbx.make_trajectory_buffer(
            add_batch_size=self.config.batch_size,
            sample_batch_size=self.config.batch_size,
            sample_sequence_length=self.config.rollout_length,
            period=0,
            min_length_time_axis=self.config.batch_size
        )
        self.buffer_fn.add()
        buffer_state = self.buffer_fn.init(self.config.transition_template)

        # elif self.config.buffer_type == "PER":
        #     raise Exception("PER buffers have not been added yet.")
        # else:
        #     raise Exception("Unknown buffer type")


        return buffer_state

    def _init_optimizer(self, optimizer_params: OptimizerParams) -> optax.chain:
        """
        Optimizer initialization. This method uses the optax optimizer function given in the agent configuration to
        initialize the appropriate optimizer. In this way, the optimizer can be initialized within the "train" method,
        and thus several combinations of its parameters can be ran with jax.vmap.
        Jitting is neither possible nor necessary, since the method returns an optax.chain class, not numerical output.
        :param optimizer_params: A NamedTuple containing the parametrization of the optimizer.
        :return: An optimizer in optax.chain.
        """

        optimizer_params_dict = optimizer_params._asdict()  # Transform from NamedTuple to dict
        optimizer_params_dict.pop('grad_clip', None)  # Remove 'grad_clip', since it is not part of the optimizer args.


        """
        Get dictionary of optimizer parameters to pass in optimizer. The procedure preserves parameters that:
            - are given in the OptimizerParams NamedTuple and are requested as args by the optimizer
            - are requested as args by the optimizer and are given in the OptimizerParams NamedTuple
        """

        optimizer_arg_names = self.config.optimizer.__code__.co_varnames  # List names of args of optimizer.

        # Keep only the optimizer arg names that are also part of the OptimizerParams (dict from NamedTuple)
        optimizer_arg_names = [
            arg_name for arg_name in optimizer_arg_names if arg_name in list(optimizer_params_dict.keys())
        ]
        if len(optimizer_arg_names) == 0:
            raise Exception("The defined optimizer parameters do not include relevant arguments for this optimizer."
                            "The optimizer has not been implemented yet. Define your own OptimizerParams object.")

        # Keep only the optimizer params that are arg names for the specific optimizer
        optimizer_params_dict = {arg_name: optimizer_params_dict[arg_name] for arg_name in optimizer_arg_names}

        # No need to scale by -1.0. 'TrainState.apply_gradients' is used for training, which subtracts the update.
        tx = optax.chain(
            optax.clip_by_global_norm(optimizer_params.grad_clip),
            self.config.optimizer(**optimizer_params_dict)
        )

        return tx

    # @partial(jax.jit, static_argnums=(0,))
    def _init_network(self, rng: PRNGKeyArray, network:Type[flax.linen.Module])\
            -> Tuple[PRNGKeyArray, Union[Dict, FrozenDict]]:
        """
        Initialization of the policy network (Q-model as a Neural Network).
        :param rng: Random key for initialization.
        :return: A random key after splitting the input and the initial parameters of the policy network.
        """

        network = network(self.n_actions, self.config)

        rng, *_rng = jax.random.split(rng, 3)
        dummy_reset_rng, network_init_rng = _rng

        dummy_state, _ = self.env.reset(dummy_reset_rng, self.env_params)
        init_x = jnp.zeros((1, dummy_state.size))

        params = network.init(network_init_rng, init_x)

        return network, params

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, rng: PRNGKeyArray) -> Tuple[PRNGKeyArray, Float[Array, "state_size"], Type[LogEnvState]]:
        """
        Environment reset.
        :param rng: Random key for initialization.
        :return: A random key after splitting the input, the reset environment in array and LogEnvState formats.
        """

        rng, reset_rng = jax.random.split(rng)
        state, env_state = self.env.reset(reset_rng, self.env_params)
        return rng, state, env_state

    @partial(jax.jit, static_argnums=(0,))
    def _env_step(self, rng: PRNGKeyArray, env_state: Type[NamedTuple], action: Int[Array, "1"]) ->\
        Tuple[PRNGKeyArray, Float[Array, "state_size"], Type[LogEnvState], Float[Array, "1"], Bool[Array, "1"], Dict]:
        """
        Environment step.
        :param rng: Random key for initialization.
        :param env_state: The environment state in LogEnvState format.
        :param action: The action selected by the agent.
        :return: A tuple of: a random key after splitting the input, the next state in array and LogEnvState formats,
                 the collected reward after executing the action, episode termination and a dictionary of optional
                 additional information.
        """

        rng, step_rng = jax.random.split(rng)
        next_state, next_env_state, reward, terminated, info =\
            self.env.step(step_rng, env_state, action.squeeze(), self.env_params)

        return rng, next_state, next_env_state, reward, terminated, info


    """ METHODS FOR TRAINING """

    @partial(jax.jit, static_argnums=(0,))
    def _make_transition(self,
                         state: Float[Array, "state_size"],
                         action: Int[Array, "1"],
                         value: Float[Array, "1"],
                         log_prob: Float[Array, "1"],
                         reward: Float[Array, "1"],
                         next_state: Float[Array, "state_size"],
                         terminated: Bool[Array, "1"],
                         info: Dict) -> Transition:
        """
        Creates a transition object based on the input and output of an episode step.
        :param state: The current state of the episode step in array format.
        :param action: The action selected by the agent.
        :param value: The critic value of the selected action.
        :param log_prob: The actor log-probability of the selected action.
        :param reward: The collected reward after executing the action.
        :param next_state: The next state of the episode step in array format.
        :param terminated: Episode termination.
        :param info: Dictionary of optional additional information.
        :return: A transition object storing information about the state before and after executing the episode step,
                 the executed action, the collected reward, episode termination and optional additional information.
        """

        transition = Transition(state.squeeze(),
                                action,
                                value,
                                log_prob,
                                # jnp.expand_dims(reward, axis=0),
                                reward,
                                next_state,
                                # jnp.expand_dims(terminated, axis=0),
                                terminated,
                                info)
        transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), transition)

        return transition

    @partial(jax.jit, static_argnums=(0,))
    def _select_action(self, rng: PRNGKeyArray, pi) -> Tuple[PRNGKeyArray, Int[Array, "1"]]:
        """
        The agent selects an action to be executed using an epsilon-greedy policy. The value of epsilon is defined by
        the "get_epsilon" method, which is based on user input, so that different epsilon-decay function can be used.
        Also, the method selects a random action using the user defined function "act_randomly", which needs to be
        passed into config during the agent's initialization. The user can modify this function so that illegal actions
        are avoided and the environment does not need to penalize the agent. Probably, this can smoothen training.
        :param rng: Random key for initialization.
        :param pi:
        :return:
        """

        rng, rng_action_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_action_sample)
        return rng, action

    @partial(jax.jit, static_argnums=(0,))
    def _rewards_to_go(self, traj_batch):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(self, update_runner: UpdateRunner, i_update_step: int) -> Tuple[UpdateRunner, Transition]:
        """

        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :param i_update_step:
        :return: Current training step. Required for printing the progressbar via jax_tqdm.
        """

        step_runner = jax.vmap(lambda v, w, x, y, z: (v, w, x, y, z), in_axes=(0, 0, None, None, 0))\
            (update_runner.env_state, update_runner.state, update_runner.actor_training, update_runner.critic_training, update_runner.rng)

        step_runner, traj_batch = jax.vmap(lambda x: lax.scan(self._step, x, None, self.config.rollout_length))(step_runner)

        env_state, state, _, _, rng = step_runner

        traj_batch = jax.tree_util.tree_map(lambda x: x.squeeze(), traj_batch)

        last_value = update_runner.critic_training.apply_fn(update_runner.critic_training.params, state)
        last_value = jnp.asarray(last_value)

        returns = self._returns(traj_batch, last_value, update_runner.hyperparams.gamma, update_runner.hyperparams.gae_lambda)

        actor_grad_fn = jax.grad(self._actor_loss, allow_int=True)
        actor_grads = actor_grad_fn(
            update_runner.actor_training,
            traj_batch.state,
            traj_batch.action,
            returns,
            traj_batch.value,
            update_runner.hyperparams.ent_coeff
        )

        critic_grad_fn = jax.grad(self._critic_loss, allow_int=True)
        critic_grads = critic_grad_fn(update_runner.critic_training, traj_batch.state, returns,
                                      update_runner.hyperparams.vf_coeff)

        actor_training = update_runner.actor_training.apply_gradients(grads=actor_grads.params)
        critic_training = update_runner.critic_training.apply_gradients(grads=critic_grads.params)

        update_runner = update_runner.replace(actor_training=actor_training, critic_training=critic_training,
                                              env_state=env_state, state=state, rng=rng)

        # metrics = self._generate_metrics(runner=update_runner, reward=traj_batch.reward, terminated=traj_batch.terminated)

        return update_runner, {}

    @partial(jax.jit, static_argnums=(0,))
    def _returns(self, traj_batch, last_value, gamma, lambda_):

        rewards_t = traj_batch.reward.squeeze()
        values_t = jnp.concatenate([traj_batch.value.squeeze(), last_value[..., jnp.newaxis]], axis=-1)[:, 1:]
        discounted_rewards_t = 1.0 - traj_batch.terminated.astype(jnp.float32).squeeze()
        discounted_rewards_t = (discounted_rewards_t * gamma).astype(jnp.float32)

        rewards_t, discounted_rewards_t, values_t = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (rewards_t, discounted_rewards_t, values_t)
        )

        # import numpy as np
        # A = np.asarray(traj_batch.terminated)

        lambda_ = jnp.ones_like(discounted_rewards_t) * lambda_

        def _body(acc, xs):
            returns, discounts, values, lambda_ = xs
            acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
            return acc, acc

        _, returns = jax.lax.scan(_body, values_t[-1], (rewards_t, discounted_rewards_t, values_t, lambda_), reverse=True)

        returns = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), returns)

        return returns


    @partial(jax.jit, static_argnums=(0,))
    # def _actor_loss(self, params, apply_fn, state, action, returns, value, ent_coef):
    def _actor_loss(self, training, state, action, returns, value, ent_coef):

        actor_policy = training.apply_fn(training.params, state)
        # actor_policy = apply_fn(params, state)
        log_prob = actor_policy.log_prob(action)
        advantage = returns - value

        loss_actor = -advantage * log_prob
        entropy = actor_policy.entropy().mean()

        total_loss_actor = loss_actor.mean() - ent_coef * entropy

        return - total_loss_actor  # TODO: negative gradient, because we want ascent but 'apply_gradients' applies descent

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, training, state, targets, vf_coef):
        value = training.apply_fn(training.params, state)
        value_loss = jnp.mean((value-targets) ** 2)
        critic_total_loss = vf_coef * value_loss
        return - critic_total_loss  # TODO: negative gradient, because we want ascent but 'apply_gradients' applies descent

    @partial(jax.jit, static_argnums=(0,))
    def _update_epoch(self, update_runner, i_update_epoch):

        runner, advantages, targets, traj_batch = update_runner

        grad_fn = jax.jit(jax.grad(self._loss, has_aux=False, allow_int=True, argnums=0))
        grads = grad_fn(
            runner.training.params,
            traj_batch,
            advantages,
            targets,
            runner.hyperparams
        )

        training = runner.training.apply_gradients(grads=grads)

        """Update runner as a dataclass"""
        runner = runner.replace(training=training)

        update_runner = (runner, advantages, targets, traj_batch)

        return update_runner, {}

    @partial(jax.jit, static_argnums=(0,))
    def _gae(self, carry, transition):
        gae, next_value, gamma, gae_lambda = carry
        terminated, value, reward = (
            transition.terminated,
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value * (1 - terminated) - value
        gae = (delta + gamma * gae_lambda * (1 - terminated) * gae)
        return (gae.squeeze(), value.squeeze(), gamma, gae_lambda), gae

    @partial(jax.jit, static_argnums=(0,))
    def _advantage(self, traj_batch, last_value, hyperparams):

        carry = (jnp.zeros_like(last_value), last_value, hyperparams.gamma, hyperparams.gae_lambda)
        _, advantages = lax.scan(self._gae, carry, traj_batch, reverse=True)

        return advantages, advantages + traj_batch.value

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, step_runner: UpdateRunner, i_step: int) -> Tuple[UpdateRunner, Transition]:
        """
        Performs an episode step. This includes:
        - The agent selecting an action.
        - Performing an environment step using this action and the current state of the environment.
        - Creating a transition based on the input and output of the environment step and storing it in the agent's
          buffer.
        - Updating the policy network.
        - Updating the target network.
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :param i_step: Current step in trajectory sampling.
        :return: A tuple containing:
                 - the step runner object, updated after performing an episode step.
                 - a dictionary of metrics regarding episode evolution and user-defined metrics.
        """

        env_state, state, actor_training, critic_training, rng = step_runner

        pi, value = self._pi_value(actor_training, critic_training, state)

        rng, action = self._select_action(rng, pi)

        log_prob = pi.log_prob(action)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, env_state, action)

        step_runner = (next_env_state, next_state, actor_training, critic_training, rng)

        transition = self._make_transition(state, action, value, log_prob, reward, next_state, terminated, info)

        return step_runner, transition

    @partial(jax.jit, static_argnums=(0,))
    def _generate_metrics(self, runner: UpdateRunner, reward: Float[Array, "1"], terminated: Bool[Array, "1"]) -> Dict:
        """
        Generate metrics of performed step, which is accumulated over steps and passed as output via lax.scan. The
        metrics include at least: episode termination and the collected reward in step. Upon the user's request, this
        method can also return the performance of the agent, based on an input function given by the user in the
        configuration of the agent during initialization, and a return of the policy network parameters.
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :param reward: The collected reward after executing the action.
        :param terminated: Episode termination.
        :return: A dictionary of step metrics.
        """

        metric = {
            "done": terminated.flatten(),
            "reward": reward.flatten()
        }

        if self.config.store_agent:
            network_params = {"network_params": jax.tree_util.tree_leaves(runner.training.params)}
            metric.update(network_params)

        if self.config.get_performance is not None:
            performance_metric = {"performance": self.config.get_performance(runner.training.step, runner)}
            metric.update(performance_metric)

        return metric

    # @partial(jax.jit, static_argnums=(0,))
    def _create_training(self, rng: PRNGKeyArray, network: Type[flax.linen.Module], optimizer_params: OptimizerParams)\
            -> TrainState:
        """

        :param rng:
        :param optimizer_params:
        :return:
        """

        network, params = self._init_network(rng, network)
        tx = self._init_optimizer(optimizer_params)
        return TrainState.create(apply_fn=network.apply, tx=tx, params=params)

    @partial(jax.jit, static_argnums=(0,))
    def _create_update_runner(self, rng: PRNGKeyArray, actor_training, critic_training, hyperparams) -> UpdateRunner:
        """

        :param rng:
        :param actor_training:
        :param critic_training:
        :param hyperparams:
        :return:
        """

        rng, reset_rng, runner_rng = jax.random.split(rng, 3)
        reset_rngs = jax.random.split(reset_rng, self.config.batch_size)
        runner_rngs = jax.random.split(runner_rng, self.config.batch_size)

        # _, state, env_state = self._reset(reset_rng)
        _, state, env_state = jax.vmap(self._reset)(reset_rngs)

        update_runner = UpdateRunner(actor_training, critic_training, env_state, state, runner_rngs, hyperparams)

        return update_runner

    @partial(jax.jit, static_argnums=(0,))
    def train(self, rng: PRNGKeyArray, hyperparams: HyperParameters) -> Tuple[UpdateRunner, Dict]:
        """
        Trains the agent. A jax_tqdm progressbar has been added in the lax.scan loop.
        :param rng: Random key for initialization. This is the original key for training.
        :param hyperparams: A set of hyperparameters for training the agent as one of the following objects:
                            HyperParameters, CategoricalHyperParameters, QuantileHyperParameters. For more information
                            on these objects check dqn_utils. The selected object must match the agent.
        :return: The final state of the step runner after training and the training metrics accumulated over all
                 training steps.
        """

        rng, *_rng = jax.random.split(rng, 4)
        actor_init_rng, critic_init_rng, runner_rng = _rng

        actor_training = self._create_training(
            actor_init_rng, self.config.actor_network, hyperparams.actor_optimizer_params
        )
        critic_training = self._create_training(
            critic_init_rng, self.config.critic_network,  hyperparams.critic_optimizer_params
        )

        update_runner = self._create_update_runner(runner_rng, actor_training, critic_training, hyperparams)

        runner, metrics = lax.scan(
            scan_tqdm(self.config.n_steps)(self._update_step),
            update_runner,
            jnp.arange(self.config.n_steps),
            self.config.n_steps
        )

        return runner, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _pi_value(self, actor_training: Dict, critic_training: Dict, state: Float[Array, "state_size"])\
            -> Float[Array, "batch_size"]:
        """
        Placeholder for agent-specific method for calculating the state-action (Q) values for a state using the policy
        network.
        :param params: Parameter of the policy network.
        :param state: State where the state-action values will be calculated.
        :return: State-action values for the input state.
        """

        pi = actor_training.apply_fn(lax.stop_gradient(actor_training.params), state)
        value = critic_training.apply_fn(lax.stop_gradient(critic_training.params), state)
        return pi, value

    """ METHODS FOR APPLYING AGENT"""

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, state: Float[Array, "state_size"]) -> Float[Array, "batch_size"]:
        """
        Calculates the action of the optimal policy for a state using the policy network parameters defined as optimal
        in post-processing (by the parent class).
        :param state: State where the state-action values will be calculated.
        :return: The action of the optimal policy.
        """

        if self.agent_trained:
            return jnp.argmax(self._q(self.agent_params, state), axis=jnp.max(state.shape))
        else:
            raise Exception("The agent has not been trained.")


    """ METHODS FOR PERFORMANCE EVALUATION """

    @partial(jax.jit, static_argnums=(0,))
    def _eval_step(self, runner: EvalRunner, i_step: int) -> Tuple[UpdateRunner, Dict]:
        """
        Performs an episode step for evaluation. This includes:
        - The agent selecting an action based on the trained policy network.
        - Performing an environment step using this action and the current state of the environment.
        - Generating metrics regarding the step.
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :param i_step: Current training step. Required for printing the progressbar via jax_tqdm.
        :return: A tuple containing:
                 - the step runner object, updated after performing an episode step.
                 - a dictionary of metrics regarding episode evolution and user-defined metrics.
        """

        pi, value = self._pi_value(self.actor_training, self.critic_training, runner.state)
        rng, action = self._select_action(runner.rng, pi)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, runner.env_state, action)

        """Update runner as a dataclass"""
        runner = runner.replace(rng=rng, state=next_state, env_state=next_env_state)

        metrics = {"done": terminated, "reward": reward}

        return runner, metrics

    def eval(self, rng: PRNGKeyArray, n_evals: int = 1e5) -> Dict:
        """
        Evaluates the trained agent's performance in the training environment. So, the performance of the agent can be
        isolated from agent training. The evaluation can be parallelized via jax.vmap.
        :param rng:  Random key for evaluation.
        :param n_evals: Number of steps in agent evaluation.
        :return: Dictionary of evaluation metrics.
        """

        rng, state, env_state = self._reset(rng)

        rng, runner_rng = jax.random.split(rng)

        runner = EvalRunner(env_state, state, rng)

        _, eval_metrics = lax.scan(
            scan_tqdm(n_evals)(self._eval_step),
            runner,
            jnp.arange(n_evals),
            n_evals
        )

        return eval_metrics


    """ METHODS FOR POST-PROCESSING """

    def collect_training(self, runner: Optional[UpdateRunner] = None, metrics: Optional[Dict] = None) -> None:
        """
        Collects training of output (the final state of the runner after training and the collected metrics).
        """

        self.agent_trained = True
        self.training_runner = runner
        self.training_metrics = metrics
        self._pp()

    def _pp(self) -> None:
        """
        Post-processes the training results,, which includes:
            - Setting the policy network parameters to be the parameters of the runner #TODO: Update for stored agent.
            - Extracting the buffer from the runner so that the training history can be exported.
        :return:
        """

        self.actor_training = self.training_runner.actor_training
        self.critic_training = self.training_runner.actor_training

    @staticmethod
    def _summary_stats(episode_metric: np.ndarray["size_metrics", float]) -> MetricStats:
        """
        Summarizes statistics for sample of episode metric (to be used for training or evaluation).
        :param episode_metric: Metric collected in training or evaluation adjusted for each episode.
        :return: Summary of episode metric.
        """

        return MetricStats(
            episode_metric=episode_metric,
            mean=episode_metric.mean(),
            var=episode_metric.var(),
            std=episode_metric.std(),
            min=episode_metric.min(),
            max=episode_metric.max(),
            median=np.median(episode_metric),
            has_nans=np.any(np.isnan(episode_metric)),
        )

    def summarize(self, dones: Union[np.ndarray["size_metrics", bool], Bool[Array, "dim5"]],
                        metric: Union[np.ndarray["size_metrics", float], Float[Array, "dim5"]])\
            -> MetricStats:
        """
        Adjusts metric per episode and summarizes (to be used for training or evaluation).
        :param dones: Whether an episode has terminated in each step.
        :param metric: Metric per step.
        :return: Summary of metric per episode.
        """

        if not isinstance(dones, np.ndarray):
            dones = np.asarray(dones).astype(np.bool_)

        if not isinstance(metric, np.ndarray):
            metric = np.asarray(metric).astype(np.float32)

        last_done = np.where(dones)[0].max()
        episodes = np.cumsum(dones[:last_done])
        episodes = np.append(0, episodes)

        metric = metric[:last_done + 1]

        episode_metric = np.array([np.sum(metric[episodes == i]) for i in np.arange(episodes.max() + 1)])

        return self._summary_stats(episode_metric)


if __name__ == "__main__":
    pass
