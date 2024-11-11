"""
Implementation of the PPO agent in JAX.

Author: Antonis Mavritsakis
@Github: amavrits

"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax_tqdm import scan_tqdm
import distrax
import optax
from flax.core import FrozenDict
from jaxagents.agent_utils.ppo_utils import *
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

STATE_TYPE = Float[Array, "state_size"]
STEP_RUNNER_TYPE = Tuple[LogEnvState, STATE_TYPE, TrainState, TrainState, PRNGKeyArray]
PI_DIST_TYPE = distrax.Categorical
RETURNS_TYPE = Float[Array, "batch_size n_rollout"]


class PPOAgentBase(ABC):
    """
    Base for PPO agents.
    Can be used for both discrete or continuous action environments, and its use depends on the provided actor network.
    Follows the instructions of: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    Uses lax.scan for rollout, so trajectories may be truncated.

    Training relies on jitting several methods by treating the 'self' arg as static. According to suggested practice,
    this can prove dangerous (https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods -
    How to use jit with methods?); if attrs of 'self' change during training, the changes will not be registered in
    jit. In this case, neither agent training nor evaluation change any 'self' attrs, so using Strategy 2 of the
    suggested practice is valid. Otherwise, strategy 3 should have been used.
    """

    agent_trained: ClassVar[bool] = False  # Whether the agent has been trained.
    training_runner: ClassVar[Optional[Runner]] = None  # Runner object after training.
    training_metrics: ClassVar[Optional[Dict]] = None  # Metrics collected during training.
    eval_during_training: ClassVar[bool] = False  # Whether the agent's performance is evaluated during training

    def __init__(self, env: Type[Environment], env_params: EnvParams, config: AgentConfig) -> None:
        """
        :param env: A gymnax or custom environment that inherits from the basic gymnax class.
        :param env_params: A dataclass named "EnvParams" containing the parametrization of the environment.
        :param config: The configuration of the agent as and AgentConfig object (from vpf_utils).
        """

        self.config = config
        self.eval_during_training = self.config.eval_rng is not None
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

    def _init_network(self, rng: PRNGKeyArray, network: Type[flax.linen.Module])\
            -> Tuple[Type[flax.linen.Module], Union[Dict, FrozenDict]]:
        """
        Initialization of the actor or critic network.
        :param rng: Random key for initialization.
        :param network: The actor or critic network.
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
    def _env_step(self, rng: PRNGKeyArray, env_state: Type[NamedTuple], action: Union[Int[Array, "1"], int]) -> \
        Tuple[
            PRNGKeyArray, Float[Array, "state_size"], Type[LogEnvState], Union[float, Float[Array, "1"]],
            Union[bool, Bool[Array, "1"]], dict
        ]:
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
        next_state, next_env_state, reward, terminated, info = \
            self.env.step(step_rng, env_state, action.squeeze(), self.env_params)

        return rng, next_state, next_env_state, reward, terminated, info


    """ METHODS FOR TRAINING """

    @partial(jax.jit, static_argnums=(0,))
    def _make_transition(self,
                         state: STATE_TYPE,
                         action: Int[Array, "1"],
                         value: Float[Array, "1"],
                         log_prob: Float[Array, "1"],
                         reward: Float[Array, "1"],
                         next_state: STATE_TYPE,
                         terminated: Bool[Array, "1"],
                         info: Dict) -> Transition:
        """
        Creates a transition object based on the input and output of an episode step.
        :param state: The current state of the episode step in array format.
        :param action: The action selected by the agent.
        :param value: The critic value of the state.
        :param log_prob: The actor log-probability of the selected action.
        :param reward: The collected reward after executing the action.
        :param next_state: The next state of the episode step in array format.
        :param terminated: Episode termination.
        :param info: Dictionary of optional additional information.
        :return: A transition object storing information about the state before and after executing the episode step,
                 the executed action, the collected reward, episode termination and optional additional information.
        """

        transition = Transition(state.squeeze(), action, value, log_prob, reward, next_state, terminated, info)
        transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), transition)

        return transition

    @partial(jax.jit, static_argnums=(0,))
    def _generate_metrics(self, runner: Runner, update_step: int) -> Dict:
        """
        Generates metrics for on-policy learning. The agent performance during training is evaluated by running
        batch_size episodes (until termination). The selected metric is the sum of rewards collected dring the episode.
        If the user selects not to generate metrics (leading to faster training), an empty dictinary is returned.
        :param runner: The update runner object, containing information about the current status of the actor's/critic's
        training, the state of the environment and training hyperparameters.
        :param update_step: The number of the update step.
        :return: A dictionary of the sum of rewards collected over 'batch_size' episodes, or empty dictionary.
        """

        metric = {}
        if self.eval_during_training:
            sum_rewards = self._eval_agent(
                self.config.eval_rng,
                runner.actor_training,
                runner.critic_training,
                self.config.batch_size
            )
            metric.update({"episode_returns": sum_rewards})

        return metric

    def _create_training(self, rng: PRNGKeyArray, network: Type[flax.linen.Module], optimizer_params: OptimizerParams)\
            -> TrainState:
        """
         Creates a TrainState object for the actor of the critic.
        :param rng: Random key for initialization.
        :param network: The actor or critic network.
        :param optimizer_params: A NamedTuple containing the parametrization of the optimizer.
        :return: A TrainState object to be used in training the actor and cirtic networks.
        """

        network, params = self._init_network(rng, network)
        tx = self._init_optimizer(optimizer_params)
        return TrainState.create(apply_fn=network.apply, tx=tx, params=params)

    @partial(jax.jit, static_argnums=(0,))
    def _create_update_runner(self, rng: PRNGKeyArray, actor_training: TrainState, critic_training: TrainState,
                              hyperparams: HyperParameters) -> Runner:
        """
        Initializes the update runner as a Runner object. The runner contains batch_size initializations of the
        environment, which are used for sampling trajectories. The update runner has one TrainState for the actor and
        one for the critic network, so that trajectory batches are used to train the same parameters.
        :param rng: Random key for initialization.
        :param actor_training: The actor TrainState object used in training.
        :param critic_training: The critic TrainState object used in training.
        :param hyperparams: An instance of HyperParameters for training.
        :return: An update runner object to be used in trajectory sampling and training.
        """

        rng, reset_rng, runner_rng = jax.random.split(rng, 3)
        reset_rngs = jax.random.split(reset_rng, self.config.batch_size)
        runner_rngs = jax.random.split(runner_rng, self.config.batch_size)

        _, state, env_state = jax.vmap(self._reset)(reset_rngs)

        update_runner = Runner(
            actor_training=actor_training,
            critic_training=critic_training,
            env_state=env_state,
            state=state,
            rng=runner_rngs,
            hyperparams=hyperparams,
            best_actor_training={},
            best_critic_training={},
            best_performance=-jnp.inf
        )

        return update_runner

    @partial(jax.jit, static_argnums=(0,))
    def _pi(self, actor_training: TrainState, state: STATE_TYPE) -> PI_DIST_TYPE:
        """
        Assess the stochastic policy via the actor network for the state.
        :param actor_training: The actor TrainState object used in training.
        :param state: The current state of the episode step in array format.
        :return: The stochastic policy for the input state.
        """

        pi = actor_training.apply_fn(lax.stop_gradient(actor_training.params), state)
        return pi

    @partial(jax.jit, static_argnums=(0,))
    def _value(self, critic_training: TrainState, state: STATE_TYPE) -> Float[Array, "1"]:
        """
        Assess the stochastic policy via the actor network for the state.
        :param critic_training: The critic TrainState object used in training.
        :param state: The current state of the episode step in array format.
        :return: The value for the input state.
        """

        value = critic_training.apply_fn(lax.stop_gradient(critic_training.params), state)
        return value

    @partial(jax.jit, static_argnums=(0,))
    def _pi_value(self, actor_training: TrainState, critic_training: TrainState, state: STATE_TYPE)\
            -> Tuple[PI_DIST_TYPE, Float[Array, "1"]]:
        """
        Assess the stochastic policy via the actor network and the value via the critic network for the state.
        :param actor_training: The actor TrainState object used in training.
        :param critic_training: The critic TrainState object used in training.
        :param state: The current state of the episode step in array format.
        :return: A tuple of the stochastic policy and the value of the state.
        """

        pi = self._pi(actor_training, state)
        value = self._value(critic_training, state)
        return pi, value

    @partial(jax.jit, static_argnums=(0,))
    def _select_action(self, rng: PRNGKeyArray, pi: PI_DIST_TYPE) -> Tuple[PRNGKeyArray, Int[Array, "1"]]:
        """
        Select action by sampling from the stochastic policy for a state.
        :param rng: Random key for initialization.
        :param pi: The distax distribution procuded by the actor network indicating the stochastic policy for a state.
        :return: A random key after action selection and the selected action from the stochastic policy.
        """

        rng, rng_action_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_action_sample)
        return rng, action

    @partial(jax.jit, static_argnums=(0,))
    def _add_next_values(self, traj_batch: Transition, last_state: Float[Array, 'state_size'],
                         critic_training: TrainState) -> Transition:

        # last_state_value = critic_training.apply_fn(jax.lax.stop_gradient(critic_training.params), last_state)
        last_state_value_vmap = jax.vmap(critic_training.apply_fn, in_axes=(None, 0))
        last_state_value = last_state_value_vmap(jax.lax.stop_gradient(critic_training.params), last_state)

        """Remove first entry so that the next state values per step are in sync with the state rewards."""
        next_values_t = jnp.concatenate(
            [traj_batch.value.squeeze(), last_state_value[..., jnp.newaxis]],
            axis=-1)[:, 1:]

        traj_batch = traj_batch._replace(next_value=next_values_t)

        return traj_batch

    @partial(jax.jit, static_argnums=(0,))
    def _add_advantages(self, traj_batch: Transition, advantage: Float[Array, 'state_size']) -> Transition:

        traj_batch = traj_batch._replace(advantage=advantage)

        return traj_batch

    """Not used for PPO with GAE"""
    @partial(jax.jit, static_argnums=(0,))
    def _returns(self, traj_batch: Transition, last_next_state_value: Float[Array, "batch_size"], gamma: float,
                 gae_lambda: float) -> RETURNS_TYPE:
        """
        Calculates the returns of every step in the trajectory batch. To do so, it identifies episodes in the
        trajectories. Note that because lax.scan is used in sampling trajectories, they do not necessarily finish with
        episode termination (episodes may be truncated). Also, since the environment is not re-initialized per sampling
        step, trajectories do not start at the initial state.
        :param traj_batch: The batch of trajectories.
        :param last_next_state_value: The value of the last next state in each trajectory.
        :param gamma: Discount factor
        :param gae_lambda: The GAE λ factor.
        :return: The returns over the episodes of the trajectory batch.
        """

        rewards_t = traj_batch.reward.squeeze()
        terminated_t = 1.0 - traj_batch.terminated.astype(jnp.float32).squeeze()
        discounts_t = (terminated_t * gamma).astype(jnp.float32)

        """Remove first entry so that the next state values per step are in sync with the state rewards."""
        next_state_values_t = jnp.concatenate(
            [traj_batch.value.squeeze(), last_next_state_value[..., jnp.newaxis]],
            axis=-1)[:, 1:]

        rewards_t, discounts_t, next_state_values_t = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (rewards_t, discounts_t, next_state_values_t)
        )

        gae_lambda = jnp.ones_like(discounts_t) * gae_lambda

        traj_runner = (rewards_t, discounts_t, next_state_values_t, gae_lambda)
        end_value = jnp.take(next_state_values_t, -1, axis=0)  # Start from end of trajectory and work in reverse.
        _, returns = jax.lax.scan(self._trajectory_returns, end_value, traj_runner, reverse=True)

        returns = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), returns)

        return returns

    @partial(jax.jit, static_argnums=(0,))
    def _advantages(self, traj_batch: Transition, gamma: float, gae_lambda: float) -> RETURNS_TYPE:
        """
        Calculates the advantage of every step in the trajectory batch. To do so, it identifies episodes in the
        trajectories. Note that because lax.scan is used in sampling trajectories, they do not necessarily finish with
        episode termination (episodes may be truncated). Also, since the environment is not re-initialized per sampling
        step, trajectories do not start at the initial state.
        :param traj_batch: The batch of trajectories.
        :param last_next_state_value: The value of the last next state in each trajectory.
        :param gamma: Discount factor
        :param gae_lambda: The GAE λ factor.
        :return: The returns over the episodes of the trajectory batch.
        """

        rewards_t = traj_batch.reward.squeeze()
        values_t = traj_batch.value.squeeze()
        terminated_t = traj_batch.terminated.squeeze()
        next_state_values_t = traj_batch.next_value.squeeze()
        gamma_t = jnp.ones_like(terminated_t) * gamma
        gae_lambda_t = jnp.ones_like(terminated_t) * gae_lambda

        rewards_t, values_t, next_state_values_t, terminated_t, gamma_t, gae_lambda_t = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1),
            (rewards_t, values_t, next_state_values_t, terminated_t, gamma_t, gae_lambda_t)
        )

        traj_runner = (rewards_t, values_t, next_state_values_t, terminated_t, gamma_t, gae_lambda_t)
        """
        TODO:
        Advantage of last step is taken from the critic, in contrast to traditional approaches, where the rollout 
        ends with episode termination and the advantage is zero. Training is still successful and the influence of this
        implementation choice is negligible.
        """
        end_advantage = jnp.zeros(self.config.batch_size)
        _, advantages = jax.lax.scan(self._trajectory_advantages, end_advantage, traj_runner, reverse=True)

        advantages = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), advantages)

        return advantages

    @partial(jax.jit, static_argnums=(0,))
    def _make_step_runners(self, update_runner: Runner) -> tuple:
        """
        Creates a step_runners tuple to be used in rollout by combining the batched environments in the update_runner
        object and broadcasting the TrainState object for the critic and the network in the update_runner object to the
        same dimension.
        :param update_runner: The Runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :return: Tuple with step runners to be used in rollout.
        """

        step_runner = (
            update_runner.env_state,
            update_runner.state,
            update_runner.actor_training,
            update_runner.critic_training,
            update_runner.rng
        )
        step_runners = jax.vmap(lambda v, w, x, y, z: (v, w, x, y, z), in_axes=(0, 0, None, None, 0))(*step_runner)
        return step_runners

    @partial(jax.jit, static_argnums=(0,))
    def _rollout(self, step_runner: STEP_RUNNER_TYPE, i_step: int) -> Tuple[STEP_RUNNER_TYPE, Transition]:
        """
        Evaluation of trajectory rollout. In each step the agent:
        - evaluates policy and value
        - selects action
        - performs environment step
        - creates step transition
        :param step_runner: A tuple containing information on the environment state, the actor and critic training
        (parameters and networks) and a random key.
        :param i_step: Unused, required for lax.scan.
        :return: The updated step_runner tuple and the rollout step transition.
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
    def _process_trajectory(self, update_runner: Runner, traj_batch: Transition, last_state: STATE_TYPE) -> Transition:
        """
        Estimates the value and advantages for a batch of trajectories. For the last state of trajectory, which is not
        guaranteed to end with termination, the value is estimated using the critic network. This assumption has been
        shown to have no influence by the end of training.
        :param update_runner: The Runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param traj_batch: The batch of trajectories, as collected by in rollout.
        :param last_state: The state at the end of every trajectory in the batch.
        :return: A batch of trajectories that includes an estimate of values and advantages.
        """

        traj_batch = jax.tree_util.tree_map(lambda x: x.squeeze(), traj_batch)
        traj_batch = self._add_next_values(traj_batch, last_state, update_runner.critic_training)

        advantages = self._advantages(traj_batch, update_runner.hyperparams.gamma, update_runner.hyperparams.gae_lambda)
        traj_batch = self._add_advantages(traj_batch, advantages)

        return traj_batch

    @partial(jax.jit, static_argnums=(0,))
    def _actor_epoch(self, epoch_runner: tuple) -> tuple:
        """
        Performs a Gradient Ascent update of the actor.
        :param epoch_runner: A tuple containing the following information about the update:
        - actor_training: TrainState object for actor training
        - actor_loss_input: Tuple with the inputs required by the actor loss function.
        - kl: The KL divergence collected during the update (used in checking for early stopping).
        - epoch: The number of the current training epoch.
        - kl_threshold: The KL divergence threshold for early stopping.
        :return: The updated epoch runner.
        """

        actor_training, actor_loss_input, kl, epoch, kl_threshold = epoch_runner
        grad_input = (actor_training, *actor_loss_input)
        grad_fn = jax.grad(self._actor_loss, has_aux=True, allow_int=True)
        grads, kl = grad_fn(*grad_input)
        actor_training = actor_training.apply_gradients(grads=grads.params)

        return actor_training, actor_loss_input, kl, epoch+1, kl_threshold

    @partial(jax.jit, static_argnums=(0,))
    def _actor_training_cond(self, epoch_runner: tuple) -> Bool[Array, "1"]:
        """
        Checks whether the lax.while_loop over epochs should be terminated (either because the number of epochs has been
        met or due to KL divergence early stopping).
        :param epoch_runner: A tuple containing the following information about the update:
        - actor_training: TrainState object for actor training
        - actor_loss_input: Tuple with the inputs required by the actor loss function.
        - kl: The KL divergence collected during the update (used in checking for early stopping).
        - epoch: The number of the current training epoch.
        - kl_threshold: The KL-divergence threshold for early stopping.
        :return: Whether the lax.while_loop over training epochs finishes.
        """

        _, _, kl, epoch, kl_threshold = epoch_runner
        return jnp.logical_and(
            jnp.less(epoch, self.config.actor_epochs),
            jnp.less_equal(kl, kl_threshold)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _actor_update(self, update_runner: Runner, traj_batch: Transition) -> TrainState:
        """
        Prepares the input and performs Gradient Ascent for the actor network.
        :param update_runner: The Runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param traj_batch: The batch of trajectories.
        :return: The actor training object updated after actor_epochs steps of Gradient Ascent.
        """

        actor_loss_input = self._actor_loss_input(update_runner, traj_batch)

        start_kl, start_epoch = -jnp.inf, 1
        actor_epoch_runner = (
            update_runner.actor_training,
            actor_loss_input,
            start_kl,
            start_epoch,
            update_runner.hyperparams.kl_threshold
        )
        actor_epoch_runner = lax.while_loop(self._actor_training_cond, self._actor_epoch, actor_epoch_runner)
        actor_training, _, _, _, _ = actor_epoch_runner

        return actor_training

    @partial(jax.jit, static_argnums=(0,))
    def _critic_epoch(self, i_epoch: int, epoch_runner: tuple) -> tuple:
        """
        Performs a Gradient Descent update of the critic.
        :param: i_epoch: The current training epoch (unused but required by lax.fori_loop).
        :param epoch_runner: A tuple containing the following information about the update:
        - critic_training: TrainState object for critic training
        - critic_loss_input: Tuple with the inputs required by the critic loss function.
        :return: The updated epoch runner.
        """

        critic_training, critic_loss_input = epoch_runner
        grad_input = (critic_training, *critic_loss_input)
        grad_fn = jax.grad(self._critic_loss, allow_int=True)
        grads = grad_fn(*grad_input)
        critic_training = critic_training.apply_gradients(grads=grads.params)
        return critic_training, critic_loss_input

    @partial(jax.jit, static_argnums=(0,))
    def _critic_update(self, update_runner: Runner, traj_batch: Transition) -> TrainState:
        """
        Prepares the input and performs Gradient Descent for the critic network.
        :param update_runner: The Runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param traj_batch: The batch of trajectories.
        :return: The critic training object updated after actor_epochs steps of Gradient Ascent.
        """

        critic_loss_input = self._critic_loss_input(update_runner, traj_batch)
        critic_epoch_runner = (update_runner.critic_training, critic_loss_input)
        critic_epoch_runner = lax.fori_loop(0, self.config.critic_epochs, self._critic_epoch, critic_epoch_runner)
        critic_training, _ = critic_epoch_runner

        return critic_training

    # @partial(jax.jit, static_argnums=(0,))
    # def _update_step(self, update_runner: Runner, i_update_step: int) -> Tuple[Runner, Dict]:
    #     """
    #     An update step of the actor and critic networks. This entails:
    #     - performing rollout for sampling a batch of trajectories.
    #     - assessing the value of the last state per trajectory using the critic.
    #     - evaluating the advantage per trajectory.
    #     - updating the actor and critic network parameters via the respective loss functions.
    #     - generating in-training performance metrics.
    #     In this approach, the update_runner already has a batch of environments initialized. The environments are not
    #     initialized in the beginning of every update step, which means that trajectories to not necessarily start from
    #     an initial state (which lead to better results when benchmarking with Cartpole-v1). Moreover, the use of lax.scan
    #     for rollout means that the trajectories do not necessarily stop with episode termination (episodes can be
    #     truncated in trajectory sampling).
    #     :param update_runner: The Runner object, containing information about the current status of the actor's/
    #     critic's training, the state of the environment and training hyperparameters.
    #     :param i_update_step: Unused, required for progressbar.
    #     :return: Tuple with updated runner and dictionary of metrics.
    #     """
    #
    #     step_runners = self._make_step_runners(update_runner)
    #     scan_rollout_fn = lambda x: lax.scan(self._rollout, x, None, self.config.rollout_length)
    #     step_runners, traj_batch = jax.vmap(scan_rollout_fn)(step_runners)
    #     last_env_state, last_state, _, _, rng = step_runners
    #     traj_batch  =self._process_trajectory(update_runner, traj_batch, last_state)
    #
    #     actor_training = self._actor_update(update_runner, traj_batch)
    #     critic_training = self._critic_update(update_runner, traj_batch)
    #
    #     """Update runner as a struct.dataclass."""
    #     update_runner = update_runner.replace(
    #         env_state=last_env_state,
    #         state=last_state,
    #         actor_training=actor_training,
    #         critic_training=critic_training,
    #         rng=rng
    #     )
    #
    #     """Evaluate agent per eval_frequency steps. If not evaluating, return empty dictionary."""
    #     metrics = self._generate_metrics(runner=update_runner, update_step=i_update_step)
    #
    #     # from flax.serialization import to_state_dict
    #     # from flax.training import orbax_utils
    #     # import orbax
    #     #
    #     # def save_array(array, i):
    #     #     A = to_state_dict(array)
    #     #     # i = jax.experimental.io_callback(ff, None, i_update_step)
    #     #
    #     #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    #     #     save_args = orbax_utils.save_args_from_target(A)
    #     #     options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=10, create=True)
    #     #     checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     #         # 'C:\\Users\\Repositories\\jax-agents\\benchmarks\\cartpole v1\\log', orbax_checkpointer, options)
    #     #         '/mnt/c/Users/Repositories/jax-agents/benchmarks/cartpole v1/log', orbax_checkpointer, options)
    #     #     checkpoint_manager.save(i, A, save_kwargs={'save_args': save_args})
    #     #     print('saved dict')
    #     #
    #     #
    #     # @jax.jit
    #     # def f(x, i):
    #     #     jax.experimental.io_callback(save_array, None, x, i)
    #     #     return None
    #     #
    #     # params = actor_training.params
    #     # x = params
    #     # A = f(x, i_update_step)
    #
    #     return update_runner, metrics

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(self, i_update_step: int, update_runner: Runner) -> Runner:
        """
        An update step of the actor and critic networks. This entails:
        - performing rollout for sampling a batch of trajectories.
        - assessing the value of the last state per trajectory using the critic.
        - evaluating the advantage per trajectory.
        - updating the actor and critic network parameters via the respective loss functions.
        - generating in-training performance metrics.
        In this approach, the update_runner already has a batch of environments initialized. The environments are not
        initialized in the beginning of every update step, which means that trajectories to not necessarily start from
        an initial state (which lead to better results when benchmarking with Cartpole-v1). Moreover, the use of lax.scan
        for rollout means that the trajectories do not necessarily stop with episode termination (episodes can be
        truncated in trajectory sampling).
        :param i_update_step: Unused, required for progressbar.
        :param update_runner: The runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :return: Tuple with updated runner and dictionary of metrics.
        """

        step_runners = self._make_step_runners(update_runner)
        scan_rollout_fn = lambda x: lax.scan(self._rollout, x, None, self.config.rollout_length)
        step_runners, traj_batch = jax.vmap(scan_rollout_fn)(step_runners)
        last_env_state, last_state, _, _, rng = step_runners
        traj_batch = self._process_trajectory(update_runner, traj_batch, last_state)

        actor_training = self._actor_update(update_runner, traj_batch)
        critic_training = self._critic_update(update_runner, traj_batch)

        """Update runner as a dataclass."""
        update_runner = update_runner.replace(
            env_state=last_env_state,
            state=last_state,
            actor_training=actor_training,
            critic_training=critic_training,
            rng=rng,
        )

        return update_runner

    @partial(jax.jit, static_argnums=(0,))
    def train(self, rng: PRNGKeyArray, hyperparams: HyperParameters) -> Tuple[Runner, Dict]:
        """
        Trains the agent. A jax_tqdm progressbar has been added in the lax.scan loop.
        :param rng: Random key for initialization. This is the original key for training.
        :param hyperparams: An instance of HyperParameters for training.
        :return: The final state of the step runner after training and the training metrics accumulated over all
                 training steps.
        """

        rng, *_rng = jax.random.split(rng, 4)
        actor_init_rng, critic_init_rng, runner_rng = _rng

        actor_training = self._create_training(
            actor_init_rng, self.config.actor_network, hyperparams.actor_optimizer_params
        )
        critic_training = self._create_training(
            critic_init_rng, self.config.critic_network, hyperparams.critic_optimizer_params
        )

        update_runner = self._create_update_runner(runner_rng, actor_training, critic_training, hyperparams)

        # runner, metrics = lax.scan(
        #     scan_tqdm(self.config.n_steps, desc='Training steps')(self._update_step),
        #     update_runner,
        #     jnp.arange(self.config.n_steps),
        #     self.config.n_steps
        # )

        def a(runner, i):
            n_training_iters = self.config.n_steps - self.config.n_steps // self.config.eval_frequency * i
            n_training_iters = jnp.clip(n_training_iters, 1, self.config.eval_frequency)
            runner = lax.fori_loop(
                0,
                n_training_iters,
                self._update_step,
                runner,
            )

            metrics = self._generate_metrics(runner=runner, update_step=i)
            # mean_returns = metrics["episode_returns"].mean()
            #
            # if jnp.greater(mean_returns, runner.best_performance):
            #     best_actor = runner.actor_training
            # else:
            #     best_actor = runner.best_actor_training
            #
            # runner = runner.replace(
            #     best_actor=best_actor,
            #     best_critic=best_critic,
            #     best_performance=max(runner.best_performance, mean_returns)
            # )

            return runner, metrics

        n_iters = self.config.n_steps // self.config.eval_frequency
        runner, metrics = lax.scan(
            scan_tqdm(n_iters, desc='Training steps')(a),
            # scan_tqdm(self.config.n_steps, self.config.eval_frequency, desc='Training steps')(a),
            update_runner,
            jnp.arange(n_iters),
            n_iters
        )

        return runner, metrics

    """Not used for PPO with GAE"""
    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_returns(self, value: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the returns per episode step over a batch of trajectories.
        :param value: The values of the steps in the trajectory according to the critic (including the one of the last
         state).
        :param traj: The trajectory batch.
        :return: An array of returns.
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_advantages(self, value: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the advantages per episode step over a batch of trajectories.
        :param value: The values of the steps in the trajectory according to the critic (including the one of the last
         state).
        :param traj: The trajectory batch.
        :return: An array of returns.
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, training: TrainState, state: Float[Array, "n_rollout batch_size state_size"],
                    action: Float[Array, "n_rollout batch_size"], log_prob_old: Float[Array, "n_rollout batch_size"],
                    advantage: RETURNS_TYPE, hyperparams: HyperParameters) \
            -> Tuple[Union[float, Float[Array, "1"]], Float[Array, "1"]]:
        """
        Calculates the actor loss. For the REINFORCE agent, the advantage function is the difference between the
        discounted returns and the value as estimated by the critic.
        :param training: The actor TrainState object.
        :param state: The states in the trajectory batch.
        :param action: The actions in the trajectory batch.
        :param log_prob_old: Log-probabilities of the old policy collected over the trajectory batch.
        :param advantage: The advantage over the trajectory batch.
        :param hyperparams: The HyperParameters object used for training.
        :return: A tuple containing the actor loss and the KL divergence (for early checking stopping criterion).
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, training: TrainState, state: Float[Array, "n_rollout batch_size state_size"],
                     targets: Float[Array, "batch_size n_rollout"], hyperparams: HyperParameters) -> float:
        """
        Calculates the critic loss.
        :param training: The critic TrainState object.
        :param state: The states in the trajectory batch.
        :param targets: The returns over the trajectory batch, which act as the targets for training the critic.
        :param hyperparams: The HyperParameters object used for training.
        :return: The critic loss.
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition) -> tuple:
        """
        Prepares the input required by the actor loss function.
        :param update_runner: The runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the actor loss function.
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner: Runner, traj_batch: Transition) -> tuple:
        """
        Prepares the input required by the critic loss function.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the critic loss function.
        """

        raise NotImplementedError

    """ METHODS FOR APPLYING AGENT"""

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, state: STATE_TYPE) -> Int[Array, "1"]:
        """
        Evaluates the action of the optimal policy (argmax) according to the trained agent for the given state.
        :param state: The current state of the episode step in array format.
        :return:
        """

        if self.agent_trained:
            pi = self._pi(self.actor_training, state)
            action = jnp.argmax(pi.logits)
            return action
        else:
            raise Exception("The agent has not been trained.")

    """ METHODS FOR PERFORMANCE EVALUATION """

    def _eval_agent(self, rng: PRNGKeyArray, actor_training: TrainState, critic_training: TrainState,
                    n_episodes: int = 1) -> Float[Array, "batch_size"]:
        """
        Evaluates the agents for n_episodes complete episodes using 'lax.while_loop'.
        :param rng: A random key used for evaluating the agent.
        :param actor_training: The actor TrainState object (either mid- or post-training).
        :param critic_training: The critic TrainState object (either mid- or post-training).
        :param n_episodes: The update_runner object used during training.
        :return: The sum of rewards collected over n_episodes episodes.
        """

        rng_eval = jax.random.split(rng, n_episodes)
        rng, state, env_state = jax.vmap(self._reset)(rng_eval)

        eval_runner = (env_state, state, actor_training, critic_training, False, 0, rng)
        eval_runners = jax.vmap(
            lambda t, u, v, w, x, y, z: (t, u, v, w, x, y, z),
            in_axes=(0, 0, None, None, None, None, 0)
        )(*eval_runner)

        eval_runner = jax.vmap(lambda x: lax.while_loop(self._eval_cond, self._eval_body, x))(eval_runners)

        _, _, _, _, _, sum_rewards, _ = eval_runner

        return sum_rewards

    @partial(jax.jit, static_argnums=(0,))
    def _eval_body(self, eval_runner: tuple) -> tuple:
        """
        A step in the episode to be used with 'lax.while_loop' for evaluation of the agent in a complete episode.
        :param eval_runner: A tuple containing information about the environment state, the actor and critic training
        states, whether the episode is terminated (for checking the condition in 'lax.while_loop'), the sum of rewards
        over the episode and a random key.
        :return: The updated eval_runner tuple.
        """

        env_state, state, actor_training, critic_training, terminated, sum_rewards, rng = eval_runner

        pi = self._pi(actor_training, state)

        action = jnp.argmax(pi.logits)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, env_state, action)

        sum_rewards += reward

        eval_runner = (next_env_state, next_state, actor_training, critic_training, terminated, sum_rewards, rng)

        return eval_runner

    @partial(jax.jit, static_argnums=(0,))
    def _eval_cond(self, eval_runner: tuple) -> Bool[Array, "1"]:
        """
        Checks whether the episode is termianted, meansing that the 'lax.while_loop' can stop.
        :param eval_runner: A tuple containing information about the environment state, the actor and critic training
        states, whether the episode is terminated (for checking the condition in 'lax.while_loop'), the sum of rewards
        over the episode and a random key.
        :return: Whether the episode is terminated, which means that the while loop must stop.
        """

        _, _, _, _, terminated, _, _ = eval_runner
        return jnp.logical_not(terminated)

    def eval(self, rng: PRNGKeyArray, n_evals: int = 32) -> Float[Array, "n_evals"]:
        """
        Evaluates the trained agent's performance post-training using the trained agent's actor and critic.
        :param rng: Random key for evaluation.
        :param n_evals: Number of steps in agent evaluation.
        :return: Dictionary of evaluation metrics.
        """

        eval_metrics = self._eval_agent(rng, self.actor_training, self.critic_training, n_evals)

        return eval_metrics

    """ METHODS FOR POST-PROCESSING """

    def collect_training(self, runner: Optional[Runner] = None, metrics: Optional[Dict] = None) -> None:
        """
        Collects training of output (the final state of the runner after training and the collected metrics).
        """

        self.agent_trained = True
        self.training_runner = runner
        self.training_metrics = metrics
        self.eval_steps_in_training = jnp.arange(1, self.config.n_steps+1, self.config.eval_frequency)
        self._pp()

    def _pp(self) -> None:
        """
        Post-processes the training results,, which includes:
            - Setting the policy network parameters to be the parameters of the runner #TODO: Update for stored agent.
            - Extracting the buffer from the runner so that the training history can be exported.
        :return:
        """

        self.actor_training = self.training_runner.actor_training
        self.critic_training = self.training_runner.critic_training

    @staticmethod
    def _summary_stats(episode_metric: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]])\
            -> MetricStats:
        """
        Summarizes statistics for sample of episode metric (to be used for training or evaluation).
        :param episode_metric: Metric collected in training or evaluation adjusted for each episode.
        :return: Summary of episode metric.
        """

        metrics = MetricStats(episode_metric)
        metrics.process()
        return metrics

    def summarize(self, metric: Union[np.ndarray["size_metrics", float], Float[Array, "size_metrics"]]) -> MetricStats:
        """
        Summarizes collection of per-episode metrics.
        :param metric: Metric per episode.
        :return: Summary of metric per episode.
        """

        return self._summary_stats(metric)


class PPOAgent(PPOAgentBase):

    """
    PPO clip agent using the GAE (PPO2) for calculating the advantage. The actor loss function standardizes the
    advantage.
    See : https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    """Not used for PPO with GAE"""
    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_returns(self, value: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the returns per episode step over a batch of trajectories.
        :param value: The values of the steps in the trajectory according to the critic (including the one of the last
        state). In the begining of the method, 'value' is the value of the state in the next step in the trajectory
        (not the reverse iteration), and after calculation it is the value of the examined state in the examined step.
        :param traj: The trajectory batch.
        :return: An array of returns.
        """
        rewards, discounts, next_state_values, gae_lambda = traj
        value = rewards + discounts * ((1 - gae_lambda) * next_state_values + gae_lambda * value)
        return value, value

    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_advantages(self, advantage: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the GAE per episode step over a batch of trajectories.
        :param advantage: The GAE advantages of the steps in the trajectory according to the critic (including the one
        of the last state). In the beginning of the method, 'advantage' is the advantage of the state in the next step
        in the trajectory (not the reverse iteration), and after calculation it is the advantage of the examined state
        in each step.
        :param traj: The trajectory batch.
        :return: An array of returns.
        """
        rewards, values, next_state_values, terminated, gamma, gae_lambda = traj
        d_t = rewards + (1 - terminated) * gamma * next_state_values - values  # Temporal difference residual at time t
        advantage = d_t + gamma * gae_lambda * (1 - terminated) * advantage
        return advantage, advantage

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, training: TrainState, state: Float[Array, "n_rollout batch_size state_size"],
                    action: Float[Array, "n_rollout batch_size"], log_prob_old: Float[Array, "n_rollout batch_size"],
                    advantage: RETURNS_TYPE, hyperparams: HyperParameters) \
            -> Tuple[Union[float, Float[Array, "1"]], Float[Array, "1"]]:
        """
        Calculates the actor loss. For the REINFORCE agent, the advantage function is the difference between the
        discounted returns and the value as estimated by the critic.
        :param training: The actor TrainState object.
        :param state: The states in the trajectory batch.
        :param action: The actions in the trajectory batch.
        :param log_prob_old: Log-probabilities of the old policy collected over the trajectory batch.
        :param advantage: The GAE over the trajectory batch.
        :param hyperparams: The HyperParameters object used for training.
        :return: A tuple containing the actor loss and the KL divergence (for early checking stopping criterion).
        """

        """ Standardize GAE, greatly improves behaviour"""
        advantage = (advantage - advantage.mean(axis=0)) / (advantage.std(axis=0) + 1e-8)

        # actor_policy = training.apply_fn(training.params, state)
        actor_policy_vmap = jax.vmap(jax.vmap(training.apply_fn, in_axes=(None, 0)), in_axes=(None, 0))
        actor_policy = actor_policy_vmap(training.params, state)
        log_prob = actor_policy.log_prob(action)
        log_policy_ratio = log_prob - log_prob_old
        policy_ratio = jnp.exp(log_policy_ratio)
        kl = jnp.sum(-log_policy_ratio)

        """
        Adopt simplified formulation of clipped policy ratio * advantage as explained in the note of:
        https://spinningup.openai.com/en/latest/algorithms/ppo.html#id2
        """
        clip = jnp.where(jnp.greater(advantage, 0), 1 + hyperparams.eps_clip, 1 - hyperparams.eps_clip)
        advantage_clip = advantage * clip

        """Actual clip calculation - not used but left here for comparison to simplified version"""
        # advantage_clip = jnp.clip(policy_ratio, 1 - hyperparams.eps_clip, 1 + hyperparams.eps_clip) * advantage

        loss_actor = jnp.minimum(policy_ratio * advantage, advantage_clip)

        entropy = actor_policy.entropy().mean()
        total_loss_actor = loss_actor.mean() + hyperparams.ent_coeff * entropy

        """ Negative loss, because we want ascent but 'apply_gradients' applies descent """
        return -total_loss_actor, kl

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, training: TrainState, state: Float[Array, "n_rollout batch_size state_size"],
                     targets: RETURNS_TYPE, hyperparams: HyperParameters) -> Float[Array, "1"]:
        """
        Calculates the critic loss.
        :param training: The critic TrainState object.
        :param state: The states in the trajectory batch.
        :param targets: The targets over the trajectory batch for training the critic.
        :param hyperparams: The HyperParameters object used for training.
        :return: The critic loss.
        """

        # value = training.apply_fn(training.params, state)
        value_vmap = jax.vmap(jax.vmap(training.apply_fn, in_axes=(None, 0)), in_axes=(None, 0))
        value = value_vmap(training.params, state)
        residuals = value - targets
        value_loss = jnp.mean(residuals ** 2)
        critic_total_loss = hyperparams.vf_coeff * value_loss

        return critic_total_loss

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition) -> tuple:
        """
        Prepares the input required by the actor loss function. For the REINFORCE agent, this entails the:
        - the actions collected over the trajectory batch.
        - the log-probability of the actions collected over the trajectory batch.
        - the returns over the trajectory batch.
        - the values over the trajectory batch as evaluated by the critic.
        - the training hyperparameters.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the actor loss function.
        """

        return (
            traj_batch.state,
            traj_batch.action,
            traj_batch.log_prob,
            traj_batch.advantage,
            update_runner.hyperparams
        )

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner: Runner, traj_batch: Transition) -> tuple:
        """
        Prepares the input required by the critic loss function. For the REINFORCE agent, this entails the:
        - the states collected over the trajectory batch.
        - the targets (returns = GAE + next_value) over the trajectory batch.
        - the training hyperparameters.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the critic loss function.
        """

        return (
            traj_batch.state,
            traj_batch.advantage + traj_batch.next_value,
            update_runner.hyperparams
        )


class PPOClipCriticAgent(PPOAgentBase):

    """
    PPO clip agent using the GAE (PPO2) for calculating the advantage. The actor loss function standardizes the
    advantage. The critic loss function is clipped.
    See : https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    """Not used for PPO with GAE"""
    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_returns(self, value: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the returns per episode step over a batch of trajectories.
        :param value: The values of the steps in the trajectory according to the critic (including the one of the last
        state). In the begining of the method, 'value' is the value of the state in the next step in the trajectory
        (not the reverse iteration), and after calculation it is the value of the examined state in the examined step.
        :param traj: The trajectory batch.
        :return: An array of returns.
        """
        rewards, discounts, next_state_values, gae_lambda = traj
        value = rewards + discounts * ((1 - gae_lambda) * next_state_values + gae_lambda * value)
        return value, value

    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_advantages(self, advantage: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the GAE per episode step over a batch of trajectories.
        :param advantage: The GAE advantages of the steps in the trajectory according to the critic (including the one
        of the last state). In the beginning of the method, 'advantage' is the advantage of the state in the next step
        in the trajectory (not the reverse iteration), and after calculation it is the advantage of the examined state
        in each step.
        :param traj: The trajectory batch.
        :return: An array of returns.
        """
        rewards, values, next_state_values, terminated, gamma, gae_lambda = traj
        d_t = rewards + (1 - terminated) * gamma * next_state_values - values  # Temporal difference residual at time t
        advantage = d_t + gamma * gae_lambda * (1 - terminated) * advantage
        return advantage, advantage

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, training: TrainState, state: Float[Array, "n_rollout batch_size state_size"],
                    action: Float[Array, "n_rollout batch_size"], log_prob_old: Float[Array, "n_rollout batch_size"],
                    advantage: RETURNS_TYPE, hyperparams: HyperParameters) \
            -> Tuple[Union[float, Float[Array, "1"]], Float[Array, "1"]]:
        """
        Calculates the actor loss. For the REINFORCE agent, the advantage function is the difference between the
        discounted returns and the value as estimated by the critic.
        :param training: The actor TrainState object.
        :param state: The states in the trajectory batch.
        :param action: The actions in the trajectory batch.
        :param log_prob_old: Log-probabilities of the old policy collected over the trajectory batch.
        :param advantage: The GAE over the trajectory batch.
        :param hyperparams: The HyperParameters object used for training.
        :return: A tuple containing the actor loss and the KL divergence (for early checking stopping criterion).
        """

        """ Standardize GAE, greatly improves behaviour"""
        advantage = (advantage - advantage.mean(axis=0)) / (advantage.std(axis=0) + 1e-8)

        # actor_policy = training.apply_fn(training.params, state)
        actor_policy_vmap = jax.vmap(jax.vmap(training.apply_fn, in_axes=(None, 0)), in_axes=(None, 0))
        actor_policy = actor_policy_vmap(training.params, state)
        log_prob = actor_policy.log_prob(action)
        log_policy_ratio = log_prob - log_prob_old
        policy_ratio = jnp.exp(log_policy_ratio)
        kl = jnp.sum(-log_policy_ratio)

        """
        Adopt simplified formulation of clipped policy ratio * advantage as explained in the note of:
        https://spinningup.openai.com/en/latest/algorithms/ppo.html#id2
        """
        policy_ratio_clip = jnp.where(jnp.greater(advantage, 0), 1 + hyperparams.eps_clip, 1 - hyperparams.eps_clip)
        advantage_clip = advantage * policy_ratio_clip

        """Actual clip calculation - not used but left here for comparison to simplified version"""
        # advantage_clip = jnp.clip(policy_ratio, 1 - hyperparams.eps_clip, 1 + hyperparams.eps_clip) * advantage

        loss_actor = jnp.minimum(policy_ratio * advantage, advantage_clip)

        entropy = actor_policy.entropy().mean()
        total_loss_actor = loss_actor.mean() + hyperparams.ent_coeff * entropy

        """ Negative loss, because we want ascent but 'apply_gradients' applies descent """
        return -total_loss_actor, kl

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, training: TrainState, state: Float[Array, "n_rollout batch_size state_size"],
                     value_t, targets: RETURNS_TYPE, hyperparams: HyperParameters) -> Float[Array, "1"]:
        """
        Calculates the critic loss.
        :param training: The critic TrainState object.
        :param state: The states in the trajectory batch.
        :param targets: The targets over the trajectory batch for training the critic.
        :param hyperparams: The HyperParameters object used for training.
        :return: The critic loss.
        """

        # value = training.apply_fn(training.params, state)
        value_vmap = jax.vmap(jax.vmap(training.apply_fn, in_axes=(None, 0)), in_axes=(None, 0))
        value = value_vmap(training.params, state)
        residuals = value - targets
        value_loss = residuals ** 2

        value_clipped = value_t + jnp.clip(value - value, -hyperparams.eps_clip, hyperparams.eps_clip)
        residuals_clipped = value_clipped - targets
        value_loss_clipped = residuals_clipped ** 2

        critic_total_loss = jnp.maximum(value_loss, value_loss_clipped).mean()
        critic_total_loss = hyperparams.vf_coeff * critic_total_loss

        return critic_total_loss

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition) -> tuple:
        """
        Prepares the input required by the actor loss function. For the REINFORCE agent, this entails the:
        - the actions collected over the trajectory batch.
        - the log-probability of the actions collected over the trajectory batch.
        - the returns over the trajectory batch.
        - the values over the trajectory batch as evaluated by the critic.
        - the training hyperparameters.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the actor loss function.
        """

        return (
            traj_batch.state,
            traj_batch.action,
            traj_batch.log_prob,
            traj_batch.advantage,
            update_runner.hyperparams
        )

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner: Runner, traj_batch: Transition) -> tuple:
        """
        Prepares the input required by the critic loss function. For the REINFORCE agent, this entails the:
        - the states collected over the trajectory batch.
        - the targets (returns = GAE + next_value) over the trajectory batch.
        - the training hyperparameters.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the critic loss function.
        """

        return (
            traj_batch.state,
            traj_batch.value,
            traj_batch.advantage + traj_batch.next_value,
            update_runner.hyperparams
        )


if __name__ == "__main__":
    pass
