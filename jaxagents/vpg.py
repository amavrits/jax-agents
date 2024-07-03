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

STATE_TYPE = Float[Array, "state_size"]
STEP_RUNNER_TYPE = Tuple[LogEnvState, STATE_TYPE, TrainState, TrainState, PRNGKeyArray]
PI_DIST_TYPE = distrax.Categorical
RETURNS_TYPE = Float[Array, "batch_size n_rollout"]


class PGAgentBase(ABC):
    """
    Vanilla Policy Gradient agent

    TODO

    Training relies on jitting several methods by treating the 'self' arg as static. According to suggested practice,
    this can prove dangerous (https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods -
    How to use jit with methods?); if attrs of 'self' change during training, the changes will not be registered in
    jit. In this case, neither agent training nor evaluation change any 'self' attrs, so using Strategy 2 of the
    suggested practice is valid. Otherwise, strategy 3 should have been used.
    """

    agent_trained: ClassVar[bool] = False  # Whether the agent has been trained.
    # Optimal policy network parameters after post-processing.
    training_runner: ClassVar[Optional[Runner]] = None  # Runner object after training.
    training_metrics: ClassVar[Optional[Dict]] = None  # Metrics collected during training.
    eval_during_training: ClassVar[bool] = False  # Whether the agent's performance is evaluated during training

    def __init__(self, env: Type[Environment], env_params: EnvParams, config: AgentConfig) -> None:
        """
        :param env: A gymnax or custom environment that inherits from the basic gymnax class.
        :param env_params: A dataclass named "EnvParams" containing the parametrization of the environment.
        :param config: The configuration of the agent as one of the following objects: AgentConfig,
                       CategoricalAgentConfig, QuantileAgentConfig. For more information
                       on these objects check dqn_utils. The selected object must match the agent.
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

    # @partial(jax.jit, static_argnums=(0,))
    def _init_network(self, rng: PRNGKeyArray, network: Type[flax.linen.Module])\
            -> Tuple[PRNGKeyArray, Union[Dict, FrozenDict]]:
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
        :param value: The critic value of the selected action.
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
    def _generate_metrics(self, runner: Runner) -> Dict:
        """
        # TODO
        :param runner: The update runner object, containing information about the current status of the actor's/critic's
        training, the state of the environment and training hyperparameters.
        :param reward: The collected reward after executing the action.
        :param terminated: Episode termination.
        :return: A dictionary of the sum of rewards collected over 'batch_size' episodes, or empty dictionary.
        """

        metric = {}
        if self.eval_during_training:

            def _body(eval_runner):
                env_state, state, actor_training, critic_training, terminated, sum_rewards, rng = eval_runner
                pi, value = self._pi_value(actor_training, critic_training, state)
                rng, action = self._select_action(rng, pi)
                rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, env_state, action)
                sum_rewards += reward
                eval_runner = (next_env_state, next_state, actor_training, critic_training, terminated, sum_rewards, rng)
                return eval_runner

            def _cond(eval_runner):
                _, _, _, _, terminated, _, _ = eval_runner
                return jnp.logical_not(terminated)

            rng_eval = jax.random.split(self.config.eval_rng, self.config.batch_size)
            rng, state, env_state = jax.vmap(self._reset)(rng_eval)
            eval_runner = (env_state, state, runner.actor_training, runner.critic_training, False, 0, rng)
            eval_runners = jax.vmap(
                lambda t, u, v, w, x, y, z: (t, u, v, w, x, y, z),
                in_axes=(0, 0, None, None, None, None, 0)
            )(*eval_runner)
            eval_runner = jax.vmap(lambda x: lax.while_loop(_cond, _body, x))(eval_runners)
            _, _, _, _, _, sum_rewards, _ = eval_runner
            metric.update({"episode_rewards": sum_rewards})

        return metric

    def _create_training(self, rng: PRNGKeyArray, network: Type[flax.linen.Module], optimizer_params: OptimizerParams)\
            -> TrainState:
        """
                # TODO

        :param rng: Random key for initialization.
        :param network: The actor or critic network.
        :param optimizer_params: A NamedTuple containing the parametrization of the optimizer.
        :return:
        """

        network, params = self._init_network(rng, network)
        tx = self._init_optimizer(optimizer_params)
        return TrainState.create(apply_fn=network.apply, tx=tx, params=params)

    @partial(jax.jit, static_argnums=(0,))
    def _create_update_runner(self, rng: PRNGKeyArray, actor_training: TrainState, critic_training: TrainState,
                              hyperparams: HyperParameters) -> Runner:
        """
        # TODO
        :param rng: Random key for initialization.
        :param actor_training: The actor TrainState object used in training.
        :param critic_training: The critic TrainState object used in training.
        :param hyperparams: An instance of HyperParameters for training.
        :return:
        """

        rng, reset_rng, runner_rng = jax.random.split(rng, 3)
        reset_rngs = jax.random.split(reset_rng, self.config.batch_size)
        runner_rngs = jax.random.split(runner_rng, self.config.batch_size)

        _, state, env_state = jax.vmap(self._reset)(reset_rngs)

        update_runner = Runner(actor_training, critic_training, env_state, state, runner_rngs, hyperparams)

        return update_runner

    @partial(jax.jit, static_argnums=(0,))
    def _pi_value(self, actor_training: TrainState, critic_training: TrainState, state: STATE_TYPE)\
            -> Tuple[PI_DIST_TYPE, Float[Array, "1"]]:
        """
        # TODO
        :param actor_training: The actor TrainState object used in training.
        :param critic_training: The critic TrainState object used in training.
        :param state: The current state of the episode step in array format.
        :return:
        """

        pi = actor_training.apply_fn(lax.stop_gradient(actor_training.params), state)
        value = critic_training.apply_fn(lax.stop_gradient(critic_training.params), state)
        return pi, value

    @partial(jax.jit, static_argnums=(0,))
    def _select_action(self, rng: PRNGKeyArray, pi: PI_DIST_TYPE) -> Tuple[PRNGKeyArray, Int[Array, "1"]]:
        """
                # TODO
        :param rng: Random key for initialization.
        :param pi: The distax distribution procuded by the actor network.
        :return:
        """

        rng, rng_action_sample = jax.random.split(rng)
        action = pi.sample(seed=rng_action_sample)
        return rng, action

    def _update_training(self, training: TrainState, loss_fn: Callable, loss_input: Tuple) -> TrainState:
        """
                        # TODO
        :param training: The actor or critic TrainState object used in training.
        :param loss_fn: The acrtor or critic loss function.
        :param loss_input: The input for the actor or critic loss function.
        :return:
        """
        grad_fn = jax.grad(loss_fn, allow_int=True)
        grads = grad_fn(*loss_input)
        return training.apply_gradients(grads=grads.params)

    @partial(jax.jit, static_argnums=(0,))
    def _returns(self, traj_batch: Transition, last_value: Float[Array, "1"], gamma: float, lambda_: float) \
            -> RETURNS_TYPE:
        """
                                # TODO

        :param traj_batch: The batch of trajectories.
        :param last_value: The value of the last state in each trajectory.
        :param gamma: Discount factor
        :param lambda_: The λ factor.
        :return:
        """
        rewards_t = traj_batch.reward.squeeze()
        values_t = jnp.concatenate([traj_batch.value.squeeze(), last_value[..., jnp.newaxis]], axis=-1)[:, 1:]
        discounted_rewards_t = 1.0 - traj_batch.terminated.astype(jnp.float32).squeeze()
        discounted_rewards_t = (discounted_rewards_t * gamma).astype(jnp.float32)

        rewards_t, discounted_rewards_t, values_t = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), (rewards_t, discounted_rewards_t, values_t)
        )

        lambda_ = jnp.ones_like(discounted_rewards_t) * lambda_

        traj_runner = (rewards_t, discounted_rewards_t, values_t, lambda_)
        _, returns = jax.lax.scan(self._trajectory_returns, values_t[-1], traj_runner, reverse=True)

        returns = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), returns)

        return returns

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, step_runner: STEP_RUNNER_TYPE, i_step: int) -> Tuple[STEP_RUNNER_TYPE, Transition]:
        """
                                # TODO

        :param step_runner: 
        :param i_step:
        :return:
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
    def _update_step(self, update_runner: Runner, i_update_step: int) -> Tuple[Runner, Dict]:
        """
        TODO
        TODO: explain that environment is not re-rest in every update step

        :param update_runner: The update runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param i_update_step: Unused, required for progressbar.
        :return: Tuple with updated runner and dictionary of metrics.
        """

        step_runner = (
            update_runner.env_state,
            update_runner.state,
            update_runner.actor_training,
            update_runner.critic_training,
            update_runner.rng
        )
        step_runners = jax.vmap(lambda v, w, x, y, z: (v, w, x, y, z), in_axes=(0, 0, None, None, 0))(*step_runner)

        step_runners, traj_batch = jax.vmap(lambda x: lax.scan(self._step, x, None, self.config.rollout_length))(step_runners)

        env_state, state, _, _, rng = step_runners
        traj_batch = jax.tree_util.tree_map(lambda x: x.squeeze(), traj_batch)

        last_value = update_runner.critic_training.apply_fn(update_runner.critic_training.params, state)
        last_value = jnp.asarray(last_value)

        returns = self._returns(traj_batch, last_value, update_runner.hyperparams.gamma, update_runner.hyperparams.gae_lambda)

        actor_loss_input = self._actor_loss_input(update_runner, traj_batch, returns)
        actor_training = self._update_training(update_runner.actor_training, self._actor_loss, actor_loss_input)

        critic_loss_input = self._critic_loss_input(update_runner, traj_batch, returns)
        critic_training = self._update_training(update_runner.critic_training, self._critic_loss, critic_loss_input)

        """Update runner as a dataclass"""
        update_runner = update_runner.replace(
            env_state=env_state,
            state=state,
            actor_training=actor_training,
            critic_training=critic_training,
            rng=rng,
        )

        metrics = self._generate_metrics(runner=update_runner)

        return update_runner, metrics

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

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_returns(self, value: Float[Array, "1"], traj: Transition) -> Tuple[float, float]:
        """
        #TODO
        :param value:
        :param traj:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, training: TrainState, state: Float[Array, "state_size"], action: Float[Array, "1"], 
                    returns: RETURNS_TYPE, value: Float[Array, "1"], ent_coef: float) -> float:
        """
                #TODO
        :param training:
        :param state:
        :param action:
        :param returns:
        :param value:
        :param ent_coef:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, training: TrainState, state: Float[Array, "state_size"], targets: Float[Array, "1"],
                     vf_coef: float) -> float:
        """
                #TODO
        :param training:
        :param state:
        :param targets:
        :param vf_coef:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition, returns: RETURNS_TYPE) -> tuple:
        """
                #TODO
        :param update_runner:
        :param traj_batch:
        :param returns:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner, traj_batch, returns):
        """
                #TODO
        :param update_runner:
        :param traj_batch:
        :param returns:
        :return:
        """
        raise NotImplementedError


    """ METHODS FOR APPLYING AGENT"""

    @partial(jax.jit, static_argnums=(0,))
    def policy(self, rng: PRNGKeyArray, state: STATE_TYPE) -> int:
        """

        :param rng: Random key for evaluation.
        :param state: The current state of the episode step in array format.
        :return:
        """

        if self.agent_trained:
            pi, value = self._pi_value(self.actor_training, self.critic_training, state)
            rng, action = self._select_action(rng, pi)
            return action
        else:
            raise Exception("The agent has not been trained.")


    """ METHODS FOR PERFORMANCE EVALUATION """

    @partial(jax.jit, static_argnums=(0,))
    def _eval_step(self, runner: EvalRunner, i_step: int) -> Tuple[Runner, Dict]:
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

        rng, rng_policy = jax.random.split(runner.rng)

        action = self.policy(rng_policy, runner.state)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, runner.env_state, action)

        """Update runner as a dataclass"""
        runner = runner.replace(rng=rng, state=next_state, env_state=next_env_state)

        metrics = {"done": terminated, "reward": reward}

        return runner, metrics

    def eval(self, rng: PRNGKeyArray, n_evals: int = 1e5) -> Dict:
        """
        Evaluates the trained agent's performance in the training environment. So, the performance of the agent can be
        isolated from agent training. The evaluation can be parallelized via jax.vmap.
        :param rng: Random key for evaluation.
        :param n_evals: Number of steps in agent evaluation.
        :return: Dictionary of evaluation metrics.
        """

        runner_rng, state, env_state = self._reset(rng)

        runner = EvalRunner(env_state, state, runner_rng)

        _, eval_metrics = lax.scan(
            scan_tqdm(n_evals)(self._eval_step),
            runner,
            jnp.arange(n_evals),
            n_evals
        )

        return eval_metrics


    """ METHODS FOR POST-PROCESSING """

    def collect_training(self, runner: Optional[Runner] = None, metrics: Optional[Dict] = None) -> None:
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
        self.critic_training = self.training_runner.critic_training

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


class ReinforceAgent(PGAgentBase):

    @partial(jax.jit, static_argnums=(0,))
    def _trajectory_returns(self, value: Float[Array, "1"], traj: Transition) -> Tuple[float, float]:
        rewards, discounts, values, lambda_ = traj
        value = rewards + discounts * ((1 - lambda_) * values + lambda_ * value)
        return value, value

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss(self, training: TrainState, state: Float[Array, "state_size"], action: Float[Array, "1"], 
                    returns: RETURNS_TYPE, value: Float[Array, "1"], ent_coef: float) -> float:
        actor_policy = training.apply_fn(training.params, state)
        log_prob = actor_policy.log_prob(action)
        advantage = returns - value

        """ Negative gradient, because we want ascent but 'apply_gradients' applies descent """
        loss_actor = -advantage * log_prob
        entropy = actor_policy.entropy().mean()

        total_loss_actor = loss_actor.mean() - ent_coef * entropy

        return total_loss_actor

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(self, training: TrainState, state: Float[Array, "state_size"], targets: Float[Array, "1"],
                     vf_coef: float) -> float:
        value = training.apply_fn(training.params, state)
        value_loss = jnp.mean((value - targets) ** 2)
        critic_total_loss = vf_coef * value_loss
        return critic_total_loss

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition, returns: RETURNS_TYPE) -> tuple:
        return (
            update_runner.actor_training,
            traj_batch.state,
            traj_batch.action,
            returns,
            traj_batch.value,
            update_runner.hyperparams.ent_coeff
        )

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner: Runner, traj_batch: Transition, returns: RETURNS_TYPE):
        return (
            update_runner.critic_training,
            traj_batch.state,
            returns,
            update_runner.hyperparams.vf_coeff
        )

if __name__ == "__main__":
    pass
