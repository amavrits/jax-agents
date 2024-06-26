"""
Implementation of PPO agent in JAX.

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
import flashbax as fbx
import xarray as xr
from flax.core import FrozenDict
from jaxagents.agent_utils.ppo_utils import *
from gymnax.environments.environment import Environment, EnvParams
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper, LogEnvState
from abc import abstractmethod
from functools import partial
from abc import ABC
from typing import Tuple, Dict, NamedTuple, Type, Union, Optional, ClassVar
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
import warnings

warnings.filterwarnings("ignore")

BufferStateType = fbx.trajectory_buffer.BufferState


class PPOAgent(ABC):
    """
    The base class for Deep Q-Learning agents, which employ different variations of Deep Q-Networks.

    Training relies on jitting several methods by treating the 'self' arg as static. According to suggested practice,
    this can prove dangerous (https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods -
    How to use jit with methods?); if attrs of 'self' change during training, the changes will not be registered in
    jit. In this case, neither agent training nor evaluation change any 'self' attrs, so using Strategy 2 of the
    suggested practice is valid. Otherwise, strategy 3 should have been used.
    """

    agent_trained: ClassVar[bool] = False  # Whether the agent has been trained.
    # Optimal policy network parameters after post-processing.
    agent_params: ClassVar[Optional[Union[Dict, FrozenDict]]] = None
    training_runner: ClassVar[Optional[Runner]] = None  # Runner object after training.
    training_metrics: ClassVar[Optional[Dict]] = None  # Metrics collected during training.
    buffer: ClassVar[Optional[BufferStateType]] = None  # Metrics collected during training.

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
        self._init_eps_fn(self.config.epsilon_fn_style, self.config.epsilon_params)

    def __str__(self) -> str:
        """
        Returns a string containing only the non-default field values.
        """

        output_lst = [field + ': ' + str(getattr(self.config, field)) for field in self.config._fields]
        output_lst = ['Agent configuration:'] + output_lst

        return '\n'.join(output_lst)


    """ GENERAL METHODS"""

    def _init_eps_fn(self, epsilon_type: str, epsilon_params: tuple) -> None:
        """
        Initialization of the epsilon function, allowing different forms of epsilon decays over training.
        :param epsilon_type: The type of epsilon function.
        :param epsilon_params: The parametrization of the epsilon function.
        :return:
        """

        if epsilon_type == "CONSTANT":
            self.get_epsilon = jax.jit(lambda i_step: epsilon_params[0])
        elif epsilon_type == "DECAY":
            eps_start, eps_end, eps_decay = epsilon_params
            self.get_epsilon = jax.jit(lambda i_step: eps_end + (eps_start - eps_end) * jnp.exp(-i_step / eps_decay))
        else:
            raise Exception("Unknown epsilon function.")

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

        if self.config.buffer_type == "FLAT":
            self.buffer_fn = fbx.make_flat_buffer(
                max_length=self.config.buffer_size,
                min_length=self.config.batch_size,
                sample_batch_size=self.config.batch_size,
                add_sequences=True,
                add_batch_size=None,
            )
        elif self.config.buffer_type == "PER":
            raise Exception("PER buffers have not been added yet.")
        else:
            raise Exception("Unknown buffer type")

        buffer_state = self.buffer_fn.init(self.config.transition_template)

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

    @partial(jax.jit, static_argnums=(0,))
    def _init_q_network(self, rng: PRNGKeyArray) -> Tuple[PRNGKeyArray, Union[Dict, FrozenDict]]:
        """
        Initialization of the policy network (Q-model as a Neural Network).
        :param rng: Random key for initialization.
        :return: A random key after splitting the input and the initial parameters of the policy network.
        """
        rng, dummy_reset_rng, network_init_rng = jax.random.split(rng, 3)
        self.q_network = self.config.q_network(self.n_actions, self.config)
        dummy_state, _ = self.env.reset(dummy_reset_rng, self.env_params)
        init_x = jnp.zeros((1, dummy_state.size))
        network_params = self.q_network.init(network_init_rng, init_x)
        return rng, network_params

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
                         reward: Float[Array, "1"],
                         next_state: Float[Array, "state_size"],
                         terminated: Bool[Array, "1"],
                         info: Dict) -> Transition:
        """
        Creates a transition object based on the input and output of an episode step.
        :param state: The current state of the episode step in array format.
        :param action: The action selected by the agent.
        :param reward: The collected reward after executing the action.
        :param next_state: The next state of the episode step in array format.
        :param terminated: Episode termination.
        :param info: Dictionary of optional additional information.
        :return: A transition object storing information about the state before and after executing the episode step,
                 the executed action, the collected reward, episode termination and optional additional information.
        """

        transition = Transition(state.squeeze(),
                                action,
                                jnp.expand_dims(reward, axis=0),
                                next_state,
                                jnp.expand_dims(terminated, axis=0),
                                info)
        transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), transition)

        return transition

    @partial(jax.jit, static_argnums=(0,))
    def _store_transition(self, buffer_state: BufferStateType, transition: Transition)\
            -> BufferStateType:
        """
        Stores a step transition into the agent's buffer.
        :param buffer_state: The agent's current buffer.
        :param transition: The episode step transition.
        :return: The updated buffer after storing transition.
        """

        return self.buffer_fn.add(buffer_state, transition)

    @partial(jax.jit, static_argnums=(0,))
    def _select_action(self, rng: PRNGKeyArray, state: Float[Array, "state_size"], training: TrainStateDQN,
                       i_step: int) -> Tuple[PRNGKeyArray, Int[Array, "1"]]:
        """
        The agent selects an action to be executed using an epsilon-greedy policy. The value of epsilon is defined by
        the "get_epsilon" method, which is based on user input, so that different epsilon-decay function can be used.
        Also, the method selects a random action using the user defined function "act_randomly", which needs to be
        passed into config during the agent's initialization. The user can modify this function so that illegal actions
        are avoided and the environment does not need to penalize the agent. Probably, this can smoothen training.
        :param rng: Random key for initialization.
        :param state: The current state of the episode step in array format.
        :param training: Current training state of the agent.
        :param i_step: Current step of training.
        :return: A random key after splitting the input, the action selected by the agent using the epsilon-greedy
                 policy.
        """

        rng, *_rng = jax.random.split(rng, 3)
        random_action_rng, random_number_rng = _rng

        q_state = self._q(lax.stop_gradient(training.params), state)
        policy_action = jnp.argmax(q_state, 1)

        random_action = self.config.act_randomly(random_action_rng, state, self.n_actions)

        random_number = jax.random.uniform(random_number_rng, minval=0, maxval=1, shape=(1,))
        eps = self.get_epsilon(i_step)
        exploitation = jnp.greater(random_number, eps)

        action = jnp.where(exploitation, policy_action, random_action)

        return rng, action

    @partial(jax.jit, static_argnums=(0,))
    def _update_target_network(self, runner: Runner) -> Runner:
        """
        Updates the parameters of the target network using the parameters of the policy network and the agent's
        hyperparameters. The update can be either periodic or incremental. In the former case, the policy parameters are
        copied to target parameters with fixed frequency of steps (not episodes), indicated by "target_update_param".
        In the latter, the target parameters are updated in every step with the policy parameters, but the extent of
        update in controlled by the hyperparameter "target_update_param".
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :return: A step runner object for which the parameters of the target network have been updated.
        """

        if self.config.target_update_method == "PERIODIC":

            training = runner.training.replace(target_params=optax.periodic_update(
                runner.training.params,
                runner.training.target_params,
                runner.training.step,
                runner.hyperparams.target_update_param
            ))

        elif self.config.target_update_method == "INCREMENTAL":

            training = runner.training.replace(target_params=optax.incremental_update(
                runner.training.params,
                runner.training.target_params,
                runner.hyperparams.target_update_param
            ))

        else:

            training = runner.training

        """Update runner as a dataclass"""
        runner = runner.replace(training=training)

        return runner

    @partial(jax.jit, static_argnums=(0))
    def _fake_update_network(self, runner: Runner) -> Runner:
        """
        Function for fake updating the policy network parameters. Used to enable updating via jax.lax.cond
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :return: The same step runner object as in the input.
        """

        return runner

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, runner: Runner, i_step: int) -> Tuple[Runner, Dict]:
        """
        Performs an episode step. This includes:
        - The agent selecting an action.
        - Performing an environment step using this action and the current state of the environment.
        - Creating a transition based on the input and output of the environment step and storing it in the agent's
          buffer.
        - Updating the policy network.
        - Updating the target network.
        - Generating metrics regarding the step.
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :param i_step: Current training step. Required for printing the progressbar via jax_tqdm.
        :return: A tuple containing:
                 - the step runner object, updated after performing an episode step.
                 - a dictionary of metrics regarding episode evolution and user-defined metrics.
        """

        rng, action = self._select_action(runner.rng, runner.state, runner.training, i_step)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(rng, runner.env_state, action)

        transition = self._make_transition(runner.state, action, reward, next_state, terminated, info)
        buffer_state = self._store_transition(runner.buffer_state, transition)

        """Update runner as a dataclass"""
        runner = runner.replace(rng=rng, state=next_state, env_state=next_env_state, buffer_state=buffer_state)

        runner = jax.lax.cond(self.buffer_fn.can_sample(buffer_state),
                              self._update_q_network,
                              self._fake_update_network,
                              runner)

        runner = self._update_target_network(runner)

        metric = self._generate_metrics(runner, reward, terminated)

        return runner, metric

    @partial(jax.jit, static_argnums=(0,))
    def _generate_metrics(self, runner: Runner, reward: Float[Array, "1"], terminated: Bool[Array, "1"]) -> Dict:
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
            "done": terminated,
            "reward": reward
        }

        if self.config.store_agent:
            network_params = {"network_params": jax.tree_util.tree_leaves(runner.training.params)}
            metric.update(network_params)

        if self.config.get_performance is not None:
            performance_metric = {"performance": self.config.get_performance(runner.training.step, runner)}
            metric.update(performance_metric)

        return metric

    @partial(jax.jit, static_argnums=(0,))
    def _sample_batch(self, rng: PRNGKeyArray, buffer_state: BufferStateType)\
            -> Tuple[PRNGKeyArray, fbx.trajectory_buffer.BufferSample]:
        """
        Samples a batch from the agent's buffer. The size of the batch is a static argument passed in the
        configuration of the agent during initialization. Unfortunately, the batch size could not be treated as a
        dynamic hyperparameter, which would have been convenient for tuning.
        :param rng: Random key for initialization.
        :param buffer_state: The agent's current buffer.
        :return: A random key after splitting the input, sampled batch.
        """

        rng, batch_sample_rng = jax.random.split(rng)
        batch = self.buffer_fn.sample(buffer_state, batch_sample_rng)
        batch = batch.experience
        return rng, batch

    @partial(jax.jit, static_argnums=(0,))
    def _update_q_network(self, runner: Runner) -> Runner:
        """
        Updates the parameters of the policy network. This includes:
        - Sampling a batch from the agent's buffer.
        - Calculating the gradient of the loss function.
        - Updating the policy parameters and returning the updated step runner.
        The loss function is kept abstract in this class and is implemented per agent accordingly.
        :param runner: The step runner object, containing information about the current status of the agent's training,
                       the state of the environment and training hyperparameters.
        :return: The step runner with updated policy network parameters.
        """

        rng, batch = self._sample_batch(runner.rng, runner.buffer_state)

        current_state, action, reward, next_state, terminated = (
            batch.first.state.squeeze(),
            batch.first.action.squeeze(),
            batch.first.reward.squeeze(),
            batch.second.state.squeeze(),
            batch.first.terminated.squeeze(),
        )

        grad_fn = jax.jit(jax.grad(self._loss, has_aux=False, allow_int=True, argnums=0))
        grads = grad_fn(runner.training.params,
                        runner.training.target_params,
                        current_state,
                        action,
                        reward,
                        next_state,
                        terminated,
                        runner.hyperparams
                        )

        training = runner.training.apply_gradients(grads=grads)

        runner = runner.replace(rng=rng, training=training)

        return runner

    @jax.block_until_ready
    @partial(jax.jit, static_argnums=(0,))
    def train(self, rng: PRNGKeyArray, hyperparams: HyperParameters) -> Tuple[Runner, Dict]:
        """
        Trains the agent. A jax_tqdm progressbar has been added in the lax.scan loop.
        :param rng: Random key for initialization. This is the original key for training.
        :param hyperparams: A set of hyperparameters for training the agent as one of the following objects:
                            HyperParameters, CategoricalHyperParameters, QuantileHyperParameters. For more information
                            on these objects check dqn_utils. The selected object must match the agent.
        :return: The final state of the step runner after training and the training metrics accumulated over all
                 training steps.
        """
        rng, network_params = self._init_q_network(rng)

        tx = self._init_optimizer(hyperparams.optimizer_params)

        training = TrainStateDQN.create(apply_fn=self.q_network.apply,
                                        params=network_params,
                                        target_params=network_params,
                                        tx=tx)

        buffer_state = self._init_buffer()

        rng, state, env_state = self._reset(rng)

        rng, runner_rng = jax.random.split(rng)

        runner = Runner(training, env_state, state, runner_rng, buffer_state, hyperparams)

        runner, metrics = lax.scan(
            scan_tqdm(self.config.n_steps)(self._step),
            runner,
            jnp.arange(self.config.n_steps),
            self.config.n_steps
        )

        return runner, metrics

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _q(self, params: Dict, state: Float[Array, "state_size"]) -> Float[Array, "batch_size"]:
        """
        Placeholder for agent-specific method for calculating the state-action (Q) values for a state using the policy
        network.
        :param params: Parameter of the policy network.
        :param state: State where the state-action values will be calculated.
        :return: State-action values for the input state.
        """

        pass

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _q_state_action(self, params: Dict, state: Float[Array, "state_size"], action: Int[Array, "1"])\
            -> Float[Array, "batch_size"]:
        """
        Place holder for agent-specific method for calculating the state-action (Q) value for a state and a selected
        action using the policy network.
        :param params: Parameter of the policy network.
        :param state: State where the state-action values will be calculated.
        :param action: Action for which the state-action value will be calculated
        :return: State-action value for the input state and action.
        """

        pass

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _q_target(self,
                  params: Dict,
                  target_params: Dict,
                  next_state: Float[Array, "state_size"],
                  reward: Float[Array, "1"],
                  terminated: Bool[Array, "1"],
                  gamma: float) -> Float[Array, "batch_size"]:
        """
        Place holder for agent-specific method for calculating the target state-action (Q) value for the next state of
        an episode step.
        :param params: Parameter of the policy network.
        :param target_params: Parameter of the target  network.
        :param next_state: Next state of the episode step where the target state-action values will be calculated.
        :param reward: The reward collected during the episode step.
        :param terminated: Termination of the episode during the performed step.
        :param gamma: The discount parameter of the Bellman equation.
        :return: Target action state values for the next state met after the episode step.
        """

        pass

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _loss(self,
              params: Dict,
              target_params: Dict,
              current_state: Float[Array, "state_size"],
              action: Int[Array, "1"],
              reward: Float[Array, "1"],
              next_state: Float[Array, "state_size"],
              terminated: Bool[Array, "1"],
              hyperparams: HyperParameters) -> Float[Array, "batch_size"]:
        """
        Place holder for agent-specific method for calculating the training loss of the policy network.
        :param params: Parameter of the policy network.
        :param target_params: Parameter of the target  network.
        :param current_state: State before performing the episode step for which the Bellman equation is calculated and
                              where the agent is trained.
        :param action: The action executed in the episode step.
        :param reward: The reward collected during the episode step.
        :param next_state: Next state of the episode step where the target state-action values will be calculated.
        :param terminated: Termination of the episode during the performed step.
        :param hyperparams: The training hyperparameters, as described in the "train" method.
        :return: The loss between the estimate of the state value by the policy network and the calculation of the
                 state value using the target network and Bellman's equation.
        """

        pass


    """ METHODS FOR APPLYING AGENT"""

    @partial(jax.jit, static_argnums=(0,))
    def q(self, state: Float[Array, "state_size"]) -> Float[Array, "batch_size"]:
        """
        Calculates the state-action (Q) values for a state using the policy network parameters defined as optimal in
        post-processing (by the parent class).
        :param state: State where the state-action values will be calculated.
        :return: The state-action (Q) values for the state.
        """

        if not self.agent_trained:
            raise Exception("The agent has not been trained.")
        else:
            return self._q(self.agent_params, state)

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

        q_values = self.q(runner.state)

        action = jnp.argmax(q_values)

        rng, next_state, next_env_state, reward, terminated, info = self._env_step(runner.rng, runner.env_state, action)

        """Update runner as a dataclass"""
        runner = runner.replace(env_state=next_env_state, state=next_state, rng=rng)

        metric = {"done": terminated, "reward": reward}

        return runner, metric

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

        self.agent_params = self.training_runner.training.params
        self.buffer = self.training_runner.buffer_state

    def export_buffer(self) -> xr.DataArray:
        """
        Exports the history of training steps as xarray. The history is collected from the buffer.
        :return: Training hsitory from buffer re-organized in xarray. If the buffer is not full, it returns only the
                 non-empty positions.
        """

        buffer_size = self.buffer.current_index if not self.buffer.is_full else self.buffer.experience.terminated.size

        state = np.asarray(self.buffer.experience.state.squeeze())[:buffer_size]
        action = np.asarray(self.buffer.experience.action.squeeze())[:buffer_size]
        next_state = np.asarray(self.buffer.experience.next_state.squeeze())[:buffer_size]
        reward = np.asarray(self.buffer.experience.reward.squeeze())[:buffer_size]
        terminated = np.asarray(self.buffer.experience.terminated.squeeze())[:buffer_size]
        episodes = np.cumsum(terminated)
        episodes = np.append(0, episodes[:-1])  # So that episode number changes when True is met in "terminated".
        history = np.c_[episodes, state, action, next_state, reward, terminated]

        """
        var_names = [
            variable name for episodes
            variable name for states
            variable name for actions
            variable name for next states
            variable name for rewards
            variable name for termination
            ]
        """
        var_names =\
            ["episode"] +\
            ["s_"+str(i+1) for i in range(state.shape[1])] +\
            ["action"] +\
            ["ns_"+str(i+1) for i in range(state.shape[1])] +\
            ["reward"] +\
            ["terminated"]

        if self.buffer.is_full:
            description = "Training step history taken from the last %d steps of the self.buffer." % buffer_size
        else:
            description = "Training step history taken from the first %d steps of the self.buffer." % buffer_size

        x = xr.DataArray(
            data=history,
            dims=["step", "var"],
            coords={"step": np.arange(buffer_size), "var": var_names},
            attrs={"description": description}
        )

        return x

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
