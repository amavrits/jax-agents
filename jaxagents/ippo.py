"""
Implementation of the PPO agent in JAX.

Author: Antonis Mavritsakis
@Github: amavrits

"""

import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import scan_tqdm
import optax
from flax.core import FrozenDict
from jaxagents.utils.ippo_utils import *
from jaxagents.utils.truncation_utils import *
from gymnax.environments.environment import Environment, EnvParams, EnvState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper, LogEnvState
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax
import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod
from functools import partial
from abc import ABC
from typing import Tuple, Dict, Optional, ClassVar, Annotated
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
import warnings
import os
import shutil

warnings.filterwarnings("ignore")

ObsType = Float[Array, "obs_size"]
ActionType = Int[Array, "n_actors"] | Float[Array, "n_actors"]
StepRunnerType = Tuple[LogEnvState | EnvState | TruncationEnvState, ObsType, TrainState, TrainState, PRNGKeyArray]
EvalRunnerType = Tuple[
    LogEnvState | EnvState | TruncationEnvState,
    ObsType,
    TrainState,
    Bool[Array, "1"],
    Bool[Array, "1"],
    Float[Array, "n_agents"],
    Float[Array, "n_agents"],
    PRNGKeyArray,
]
ReturnsType = Float[Array, "batch_size n_rollout"]
ActorLossInputType = Tuple[
    Annotated[ObsType, "n_minibatch"],
    Annotated[ActionType, "n_minibatch"],
    Annotated[Float[Array, "1"], "n_minibatch"],
    Annotated[Float[Array, "1"], "n_minibatch"],
    HyperParameters
]
CriticLossInputType = Tuple[
    Annotated[ObsType, "n_minibatch"],
    Annotated[Float[Array, "1"], "n_minibatch"],
    HyperParameters,
]


"""
NOTE ABOUT TERMINATION, TRUNCATION AND EPISODE COMPLETION:

An episode ends either by termination (e.g., agent fails or succeeds) or by truncation (e.g., reaching the max episode
length). In both cases, the environment returns `done = True`, and Gymnax resets the environment automatically inside
the `step` function.

However, Gymnax does not guarantee a distinction between termination and truncation, which is often useful for training
and analysis. This wrapper adds that distinction.

Specifically:
- `done` is `True` when either termination or truncation occurs (preserving Gymnax behavior).
- The `info` dictionary includes two boolean flags: `terminated` and `truncated`.
- These are used in trajectory processing (e.g., return and advantage estimation) to provide better learning signals.

This is especially valuable when episodes are long and agents benefit from periodic resets. For example, in Cartpole-v1,
the goal is to keep the pole balanced for 500 steps. If we truncate episodes at 450 steps:
- `terminated = True` only when the pole falls.
- `truncated = True` when the agent reaches 450 steps without failure.

In both cases, the environment resets, but this wrapper allows distinguishing between a failure and a timeout, which
can significantly enhance learning and performance diagnostics.
"""


class IPPOBase(ABC):
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

    # Number of actors in environment
    n_actors: int
    # Function for performing a minibatch update of the actor network.
    _actor_minibatch_fn: ClassVar[Callable[
        [Tuple[TrainState, ActorLossInputType, float]],
        Tuple[TrainState, ActorLossInputType, float]]
    ]
    # Function for performing a minibatch update of the critic network.
    _critic_minibatch_fn: ClassVar[Callable[
        [Tuple[TrainState, CriticLossInputType]],
        Tuple[TrainState, CriticLossInputType]]
    ]
    agent_trained: ClassVar[bool] = False  # Whether the agent has been trained.
    training_runner: ClassVar[Optional[Runner]] = None  # Runner object after training.
    actor_training: ClassVar[Optional[TrainState]] = None  # Actor training object.
    critic_training: ClassVar[Optional[TrainState]] = None  # Critic training object.
    training_metrics: ClassVar[Optional[Dict]] = None  # Metrics collected during training.
    eval_during_training: ClassVar[bool] = True  # Whether the agent's performance is evaluated during training
    # The maximum step reached in precious training. Zero by default for starting a new training. Will be set by
    # restoring or passing a trained agent (from serial training or restoring)
    previous_training_max_step: ClassVar[int] = 0

    def __init__(
            self,
            env: Environment,
            env_params: EnvParams,
            config: AgentConfig,
            eval_during_training: bool = True
    ) -> None:

        """
        :param env: A gymnax or custom environment that inherits from the basic gymnax class.
        :param env_params: A dataclass named "EnvParams" containing the parametrization of the environment.
        :param config: The configuration of the agent as and AgentConfig object (from vpf_utils).
        :param eval_during_training: Whether evaluation should be performed during training.
        """
        self.n_actors = env.n_actors
        self.config = config
        self.eval_during_training = eval_during_training
        self._init_checkpointer()
        self._init_env(env, env_params)

    def __str__(self) -> str:
        """
        Returns a string containing only the non-default field values.
        """

        output_lst = [field + ': ' + str(getattr(self.config, field)) for field in self.config._fields]
        output_lst = ['Agent configuration:'] + output_lst

        return '\n'.join(output_lst)
    
    """ GENERAL METHODS"""

    def _init_env(self, env: Environment, env_params: EnvParams) -> None:
        """
        Environment initialization.
        :param env: A gymnax or custom environment that inherits from the basic gymnax class.
        :param env_params: A dataclass containing the parametrization of the environment.
        :return:
        """

        env = TruncationWrapper(env, self.config.max_episode_steps)
        # env = FlattenObservationWrapper(env)
        # self.env = LogWrapper(env)
        self.env = env
        self.env_params = env_params

    def _init_checkpointer(self) -> None:
        """
        Sets whether checkpointing should be performed, decided by whether a checkpoint directory has been provided. If
        so, sets the checkpoint manager using orbax.
        :return:
        """

        self.checkpointing = self.config.checkpoint_dir is not None

        if self.checkpointing:

            if not self.config.restore_agent:

                dir_exists = os.path.exists(self.config.checkpoint_dir)
                if not dir_exists:
                    os.makedirs(self.config.checkpoint_dir)

                dir_files = [
                    file for file in os.listdir(self.config.checkpoint_dir)
                    if os.path.isdir(os.path.join(self.config.checkpoint_dir, file))
                ]
                if len(dir_files) > 0:
                    for file in dir_files:
                        file_path = os.path.join(self.config.checkpoint_dir, file)
                        shutil.rmtree(file_path)

                # Log training configuration
                with open(os.path.join(self.config.checkpoint_dir, 'training_configuration.txt'), "w") as f:
                    f.write(self.__str__())

            orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())

            options = orbax.checkpoint.CheckpointManagerOptions(
                create=True,
                step_prefix='trainingstep',
            )

            self.checkpoint_manager = orbax.checkpoint.CheckpointManager(
                self.config.checkpoint_dir,
                orbax_checkpointer,
                options
            )

        else:

            self.checkpoint_manager = None

    def _create_empty_trainstate(self, network) -> TrainState:
        """
        Creates an empty TrainState object for restoring checkpoints.
        :param network: The actor or critic network.
        :return:
        """

        rng = jax.random.PRNGKey(1)  # Just a dummy PRNGKey for initializing the networks parameters.
        network, params = self._init_network(rng, network)

        optimizer_params = OptimizerParams()  # Use the default values of the OptimizerParams object.
        tx = self._init_optimizer(optimizer_params)

        empty_training = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        return empty_training

    def restore(
            self,
            mode: str = "best",
            best_fn: Optional[Callable[[Dict[str, Float[Array, "1"]]], [int]]] = None
    ) -> None:
        """
        Restores a checkpoint (best or latest) and collects the history of metrics as assessed during training. Then,
        post-processes the restored checkpoint.
        :param mode: Determines whether the best performing or latest checkpoint should be restored.
        :param best_fn: The function that should be used in determining the best performing checkpoint.
        :return:
        """

        steps = self.checkpoint_manager.all_steps()

        # Log keys in checkpoints
        ckpt = self.checkpoint_manager.restore(steps[0])
        ckpt_keys = [key for key in list(ckpt.keys()) if key != "runner"]

        # Collect history of metrics in training. Useful for continuing training.
        metrics = {key: [None] * len(steps) for key in ckpt_keys}
        for i, step in enumerate(steps):
            ckpt = self.checkpoint_manager.restore(step)
            for key in ckpt_keys:
                metrics[key][i] = ckpt[key][jnp.newaxis, :]
        metrics = {key: jnp.concatenate(val, axis=0) for (key, val) in metrics.items()}

        if mode == "best":
            if best_fn is not None:
                step = steps[best_fn(metrics)]
            else:
                raise Exception("Function for determining best checkpoint not provided")
        elif mode == "last":
            step = self.checkpoint_manager.latest_step()
        else:
            raise Exception("Unknown method for selecting a checkpoint.")

        """
        Create an empty target for restoring the checkpoint.
        Some of the arguments come from restoring one of the ckpts.
        """

        empty_actor_training = self._create_empty_trainstate(self.config.actor_network)
        empty_critic_training = self._create_empty_trainstate(self.config.critic_network)

        # Get some state and envstate for restoring the checkpoint.
        _, obs, envstate = self.env_reset(jax.random.PRNGKey(1))

        empty_runner = Runner(
            actor_training=empty_actor_training,
            critic_training=empty_critic_training,
            envstate=envstate,
            obs=obs,
            rng=jax.random.split(jax.random.PRNGKey(1), self.config.batch_size),  # Just a dummy PRNGKey for initializing the networks parameters.
            # Hyperparams can be loaded as a dict. If training continues, new hyperparams will be provided.
            hyperparams=ckpt["runner"]["hyperparams"]
        )

        target_ckpt = {
            "runner": empty_runner,
            "terminated": jnp.zeros(metrics["terminated"].shape[1]),
            "truncated": jnp.zeros(metrics["truncated"].shape[1]),
            "final_rewards": jnp.zeros(metrics["final_rewards"].shape[1]),
            "returns": jnp.zeros(metrics["returns"].shape[1]),
        }

        ckpt = self.checkpoint_manager.restore(step, items=target_ckpt)

        self.collect_training(ckpt["runner"], metrics, previous_training_max_step=max(steps))

    def _init_optimizer(self, optimizer_params: OptimizerParams) -> optax.chain:
        """
        Optimizer initialization. This method uses the optax optimizer function given in the agent configuration to
        initialize the appropriate optimizer. In this way, the optimizer can be initialized within the "train" method,
        and thus several combinations of its parameters can be ran with jax.vmap. Jit is neither possible nor necessary.
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
            raise Exception(
                "The defined optimizer parameters do not include relevant arguments for this optimizer."
                "The optimizer has not been implemented yet. Define your own OptimizerParams object."
            )

        # Keep only the optimizer params that are arg names for the specific optimizer
        optimizer_params_dict = {arg_name: optimizer_params_dict[arg_name] for arg_name in optimizer_arg_names}

        # No need to scale by -1.0. 'TrainState.apply_gradients' is used for training, which subtracts the update.
        tx = optax.chain(
            optax.clip_by_global_norm(optimizer_params.grad_clip),
            self.config.optimizer(**optimizer_params_dict)
        )

        return tx

    def _init_network(
            self,
            rng: PRNGKeyArray,
            network: flax.linen.Module
    ) -> Tuple[flax.linen.Module, FrozenDict]:
        """
        Initialization of the actor or critic network.
        :param rng: Random key for initialization.
        :param network: The actor or critic network.
        :return: A random key after splitting the input and the initial parameters of the policy network.
        """

        network = network(self.config)

        rng, *_rng = jax.random.split(rng, 3)
        dummy_reset_rng, network_init_rng = _rng

        _, dummy_obs, _ = self.env_reset(dummy_reset_rng)
        init_x = jnp.zeros((1, dummy_obs.size))

        params = network.init(network_init_rng, init_x)

        return network, params

    @partial(jax.jit, static_argnums=(0,))
    def env_reset(self, rng: PRNGKeyArray) -> Tuple[PRNGKeyArray, ObsType, LogEnvState | EnvState | TruncationEnvState]:
        """
        Environment reset.
        :param rng: Random key for initialization.
        :return: A random key after splitting the input, the reset environment in array and LogEnvState formats.
        """

        rng, reset_rng = jax.random.split(rng)
        obs, envstate = self.env.reset(reset_rng, self.env_params)

        return rng, obs, envstate

    @partial(jax.jit, static_argnums=(0,))
    def env_step(
            self,
            rng: PRNGKeyArray,
            envstate: LogEnvState | EnvState | TruncationEnvState,
            actions: ActionType
    )-> Tuple[
        PRNGKeyArray,
        ObsType,
        LogEnvState | EnvState | TruncationEnvState,
        Float[Array, "1"],
        Bool[Array, "1"],
        Dict[str, float | bool]
    ]:
        """
        Environment step.
        :param rng: Random key for initialization.
        :param envstate: The environment state in LogEnvState format.
        :param actions: The actions selected per actor.
        :return: A tuple of: a random key after splitting the input, the next state in array and LogEnvState formats,
                 the collected reward after executing the action, episode termination and a dictionary of optional
                 additional information.
        """

        rng, step_rng = jax.random.split(rng)
        next_obs, next_envstate, reward, done, info = \
            self.env.step(step_rng, envstate, actions.squeeze(), self.env_params)

        return rng, next_obs, next_envstate, reward, done, info
    
    """ METHODS FOR TRAINING """

    @partial(jax.jit, static_argnums=(0,))
    def _make_transition(
            self,
            obs: ObsType,
            actions: ActionType,
            value: Float[Array, "n_actors"],
            log_prob: Float[Array, "n_actors"],
            reward: Float[Array, "n_actors"],
            next_obs: ObsType,
            terminated: Bool[Array, "1"],
            ) -> Transition:
        """
        Creates a transition object based on the input and output of an episode step.
        :param obs: The current state of the episode step in array format.
        :param actions: The action selected per actor.
        :param value: The value of the state per critic.
        :param log_prob: The actor log-probability of the selected action.
        :param reward: The collected reward after executing the action per actor.
        :param next_obs: The next state of the episode step in array format.
        :param terminated: Episode termination.
        :return: A transition object storing information about the state before and after executing the episode step,
                 the executed action, the collected reward, episode termination and optional additional information.
        """

        transition = Transition(obs.squeeze(), actions, value, log_prob, reward, next_obs, terminated)
        transition = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), transition)

        return transition

    @partial(jax.jit, static_argnums=(0,))
    def _generate_metrics(self, runner: Runner, update_step: int) -> Dict[str, Float[Array, "n_agents"]]:
        """
        Generates metrics for on-policy learning. The agent performance during training is evaluated by running
        n_evals episodes (until termination). If the user selects not to generate metrics (leading to faster training),
        an empty dictionary is returned.
        :param runner: The update runner object, containing information about the current status of the actor's/critic's
        training, the state of the environment and training hyperparameters.
        :param update_step: The number of the update step.
        :return: A dictionary of the sum of rewards collected over 'n_evals' episodes, or empty dictionary.
        """

        metric = {}
        if self.eval_during_training:
            metric = self._eval_agent(
                self.config.eval_rng,
                runner.actor_training,
                runner.critic_training,
                self.config.n_evals
            )

        metric.update({
            "actor_loss": runner.actor_loss,
            "critic_loss": runner.critic_loss
        })

        return metric

    def _create_training(
            self,
            rng: PRNGKeyArray,
            network: type[flax.linen.Module],
            optimizer_params: OptimizerParams
    )-> TrainState:
        """
         Creates a TrainState object for the actor or the critic.
        :param rng: Random key for initialization.
        :param network: The actor or critic network.
        :param optimizer_params: A NamedTuple containing the parametrization of the optimizer.
        :return: A TrainState object to be used in training the actor and cirtic networks.
        """

        network, params = self._init_network(rng, network)
        tx = self._init_optimizer(optimizer_params)
        return TrainState.create(apply_fn=network.apply, tx=tx, params=params)

    @partial(jax.jit, static_argnums=(0,))
    def _create_update_runner(
            self,
            rng: PRNGKeyArray,
            actor_training: TrainState,
            critic_training: TrainState,
            hyperparams: HyperParameters
    ) -> Runner:
        """
        Initializes the update runner as a Runner object. The runner contains batch_size initializations of the
        environment, which are used for sampling trajectories. The update runner has one TrainState for the actor and
        one for the critic network, so that trajectory batches are used to train the same parameters.
        :param rng: Random key for initialization.
        :param actor_training: The actor TrainState objects used in training.
        :param critic_training: The critic TrainState objects used in training.
        :param hyperparams: An instance of HyperParameters for training.
        :return: An update runner object to be used in trajectory sampling and training.
        """

        rng, reset_rng, runner_rng = jax.random.split(rng, 3)
        reset_rngs = jax.random.split(reset_rng, self.config.batch_size)
        runner_rngs = jax.random.split(runner_rng, self.config.batch_size)

        _, obs, envstate = jax.vmap(self.env_reset)(reset_rngs)

        update_runner = Runner(
            actor_training=actor_training,
            critic_training=critic_training,
            envstate=envstate,
            obs=obs,
            rng=runner_rngs,
            hyperparams=hyperparams,
            actor_loss=jnp.zeros(1),
            critic_loss=jnp.zeros(1),
        )

        return update_runner

    @partial(jax.jit, static_argnums=(0,))
    def _add_next_values(
            self,
            traj_batch: Transition,
            last_obs: ObsType,
            critic_training: TrainState
    ) -> Transition:

        last_state_value_vmap = jax.vmap(critic_training.apply_fn, in_axes=(None, 0))
        last_state_value = last_state_value_vmap(lax.stop_gradient(critic_training.params), last_obs)
        last_state_value = jnp.expand_dims(last_state_value, axis=1)

        """Remove first entry so that the next state values per step are in sync with the state rewards."""
        next_values_t = jnp.concatenate([
            traj_batch.value.squeeze(),
            last_state_value
        ], axis=1)[:, 1:, :]

        traj_batch = traj_batch._replace(next_value=next_values_t)

        return traj_batch

    @partial(jax.jit, static_argnums=(0,))
    def _add_advantages(self, traj_batch: Transition, advantage: ReturnsType) -> Transition:

        traj_batch = traj_batch._replace(advantage=advantage)

        return traj_batch

    @partial(jax.jit, static_argnums=(0,))
    def _returns(
            self,
            traj_batch: Transition,
            last_next_state_value: Float[Array, "batch_size"],
            gamma: float,
            gae_lambda: float
    ) -> ReturnsType:
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
        _, returns = lax.scan(self._trajectory_returns, end_value, traj_runner, reverse=True)

        returns = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), returns)

        return returns

    @partial(jax.jit, static_argnums=(0,))
    def _advantages(
            self,
            traj_batch: Transition,
            gamma: float,
            gae_lambda: float
    ) -> ReturnsType:
        """
        Calculates the advantage of every step in the trajectory batch. To do so, it identifies episodes in the
        trajectories. Note that because lax.scan is used in sampling trajectories, they do not necessarily finish with
        episode termination (episodes may be truncated). Also, since the environment is not re-initialized per sampling
        step, trajectories do not start at the initial state.
        :param traj_batch: The batch of trajectories.
        :param last_next_state_value: The value of the last next state in each trajectory.
        :param gamma: Discount factor
        :param gae_lambda: The GAE λ factor.
        :return: The advantages over the episodes of the trajectory batch.
        """

        rewards_t = traj_batch.reward.squeeze()
        values_t = traj_batch.value.squeeze()
        terminated_t = jnp.expand_dims(traj_batch.terminated, axis=-1)
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
        end_advantage = jnp.zeros((self.config.batch_size, self.n_actors))
        _, advantages = jax.lax.scan(self._trajectory_advantages, end_advantage, traj_runner, reverse=True)
        advantages = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), advantages)

        return advantages

    @partial(jax.jit, static_argnums=(0,))
    def _make_rollout_runners(self, update_runner: Runner) -> Tuple[StepRunnerType, ...]:
        """
        Creates a rollout_runners tuple to be used in rollout by combining the batched environments in the update_runner
        object and broadcasting the TrainState object for the critic and the network in the update_runner object to the
        same dimension.
        :param update_runner: The Runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :return: Tuple with step runners to be used in rollout.
        """

        rollout_runner = (
            update_runner.envstate,
            update_runner.obs,
            update_runner.actor_training,
            update_runner.critic_training,
            update_runner.rng,
        )
        rollout_runners = jax.vmap(
            lambda v, w, x, y, z: (v, w, x, y, z), in_axes=(0, 0, None, None, 0)
        )(*rollout_runner)
        return rollout_runners

    @partial(jax.jit, static_argnums=(0,))
    def _rollout(self, step_runner: StepRunnerType, i_step: int) -> Tuple[StepRunnerType, Transition]:
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

        envstate, obs, actor_training, critic_training, rng = step_runner

        rng, rng_action = jax.random.split(rng)
        actions = self._sample_actions(rng_action, actor_training, obs)

        values = critic_training.apply_fn(lax.stop_gradient(critic_training.params), obs)

        log_prob = self._log_prob(actor_training, lax.stop_gradient(actor_training.params), obs, actions)

        rng, next_obs, next_envstate, reward, done, info = self.env_step(rng, envstate, actions)

        step_runner = (next_envstate, next_obs, actor_training, critic_training, rng)

        terminated = info["terminated"]

        transition = self._make_transition(
            obs=obs,
            actions=actions,
            value=values,
            log_prob=log_prob,
            reward=reward,
            next_obs=next_obs,
            terminated=terminated,
        )

        return step_runner, transition

    @partial(jax.jit, static_argnums=(0,))
    def _process_trajectory(self, update_runner: Runner, traj_batch: Transition, last_obs: ObsType) -> Transition:
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
        traj_batch = self._add_next_values(traj_batch, last_obs, update_runner.critic_training)

        advantages = self._advantages(traj_batch, update_runner.hyperparams.gamma, update_runner.hyperparams.gae_lambda)
        traj_batch = self._add_advantages(traj_batch, advantages)

        return traj_batch

    @staticmethod
    def _actor_minibatch_update(
            i_minibatch: int,
            minibatch_runner: Tuple[TrainState, ActorLossInputType, float],
            grad_fn: Callable[[Any], ActorLossInputType]
    ) -> Annotated[Tuple[TrainState, ActorLossInputType, float], "n_minibatch"]:
        """
        Performs a minibatch update of the actor network. Not jitted, so that the grad_fn argument can be
        passed. This choice doesn't hurt performance. To be called using a lambda function for defining grad_fn.
        :param i_minibatch: Number of minibatch update.
        :param minibatch_runner: A tuple containing the TranState object, the loss input arguments and the KL divergence.
        :param grad_fn: The gradient function of the training loss.
        :return: Minibatch runner with an updated TrainState.
        """

        actor_training, actor_loss_input, kl = minibatch_runner
        *traj_batch, hyperparams = actor_loss_input
        traj_minibatch = jax.tree_map(lambda x: jnp.take(x, i_minibatch, axis=0), traj_batch)
        grad_input_minibatch = (actor_training, *traj_minibatch, hyperparams)
        grads, kl = grad_fn(*grad_input_minibatch)
        actor_training = actor_training.apply_gradients(grads=grads.params)
        return actor_training, actor_loss_input, kl

    @staticmethod
    def _critic_minibatch_update(
            i_minibatch: int,
            minibatch_runner: Tuple[TrainState, CriticLossInputType],
            grad_fn: Callable[[Any], CriticLossInputType]
    ) -> Tuple[TrainState, CriticLossInputType]:
        """
        Performs a minibatch update of the critic network. Not jitted, so that the grad_fn argument can be
        passed. This choice doesn't hurt performance. To be called using a lambda function for defining grad_fn.
        :param i_minibatch: Number of minibatch update.
        :param minibatch_runner: A tuple containing the TranState object and the loss input arguments.
        :param grad_fn: The gradient function of the training loss.
        :return: Minibatch runner with an updated TrainState.
        """

        critic_training, critic_loss_input = minibatch_runner
        *traj_batch, hyperparams = critic_loss_input
        traj_minibatch = jax.tree_map(lambda x: jnp.take(x, i_minibatch, axis=0), traj_batch)
        grad_input_minibatch = (critic_training, *traj_minibatch, hyperparams)
        grads = grad_fn(*grad_input_minibatch)
        critic_training = critic_training.apply_gradients(grads=grads.params)
        return critic_training, critic_loss_input

    @partial(jax.jit, static_argnums=(0,))
    def _actor_epoch(
            self,
            epoch_runner: Tuple[TrainState, ActorLossInputType, Float[Array, "1"], int, float]
    ) -> Tuple[TrainState, ActorLossInputType, Float[Array, "1"], int, float]:
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
        minibatch_runner = (actor_training, actor_loss_input, 0)
        n_minibatch_updates = self.config.batch_size // self.config.minibatch_size
        minibatch_runner = lax.fori_loop(0, n_minibatch_updates, self._actor_minibatch_fn, minibatch_runner)
        actor_training, _, kl = minibatch_runner

        return actor_training, actor_loss_input, kl, epoch+1, kl_threshold

    @partial(jax.jit, static_argnums=(0,))
    def _actor_training_cond(
            self,
            epoch_runner: Tuple[TrainState, ActorLossInputType, Float[Array, "1"], int, Float[Array, "1"]]
    ) -> Bool[Array, "1"]:
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
    def _actor_update(self, update_runner: Runner, traj_batch: Transition) -> Tuple[TrainState, Float[Array, "1"]]:
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

        actor_loss, _ = self._actor_loss(
            actor_training,
            traj_batch.obs,
            traj_batch.action,
            traj_batch.log_prob,
            traj_batch.advantage,
            update_runner.hyperparams
        )

        return actor_training, actor_loss

    @partial(jax.jit, static_argnums=(0,))
    def _critic_epoch(
            self,
            i_epoch: int,
            epoch_runner: Tuple[TrainState, CriticLossInputType]
    ) -> Tuple[TrainState, CriticLossInputType]:
        """
        Performs a Gradient Descent update of the critic.
        :param: i_epoch: The current training epoch (unused but required by lax.fori_loop).
        :param epoch_runner: A tuple containing the following information about the update:
        - critic_training: TrainState object for critic training
        - critic_loss_input: Tuple with the inputs required by the critic loss function.
        :return: The updated epoch runner.
        """

        critic_training, critic_loss_input = epoch_runner
        minibatch_runner = (critic_training, critic_loss_input)
        n_minibatch_updates = self.config.batch_size // self.config.minibatch_size
        minibatch_runner = lax.fori_loop(0, n_minibatch_updates, self._critic_minibatch_fn, minibatch_runner)
        critic_training, _ = minibatch_runner

        return critic_training, critic_loss_input

    @partial(jax.jit, static_argnums=(0,))
    def _critic_update(self, update_runner: Runner, traj_batch: Transition) ->  Tuple[TrainState, Float[Array, "1"]]:
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

        critic_targets = critic_loss_input[1].reshape(-1, self.config.rollout_length, self.n_actors)
        critic_loss = self._critic_loss(critic_training, traj_batch.obs, critic_targets, update_runner.hyperparams)

        return critic_training, critic_loss

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
        :return: The updated runner
        """

        rollout_runners = self._make_rollout_runners(update_runner)
        scan_rollout_fn = lambda x: lax.scan(self._rollout, x, None, self.config.rollout_length)
        rollout_runners, traj_batch = jax.vmap(scan_rollout_fn)(rollout_runners)
        last_envstate, last_obs, _, _, rng = rollout_runners
        traj_batch = self._process_trajectory(update_runner, traj_batch, last_obs)

        actor_training, actor_loss = self._actor_update(update_runner, traj_batch)
        critic_training, critic_loss = self._critic_update(update_runner, traj_batch)

        """Update runner as a dataclass."""
        update_runner = update_runner.replace(
            envstate=last_envstate,
            obs=last_obs,
            actor_training=actor_training,
            critic_training=critic_training,
            rng=rng,
            actor_loss=jnp.expand_dims(actor_loss, axis=-1),
            critic_loss=jnp.expand_dims(critic_loss, axis=-1)
        )

        return update_runner

    @partial(jax.jit, static_argnums=(0,))
    def _checkpoint(
            self,
            update_runner: Runner,
            metrics: Dict[str, Float[Array, "n_agents"]],
            i_training_step: int
    ) -> None:
        """
        Wraps the base checkpointing method in a Python callback.
        :param update_runner: The runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param metrics: Dictionary of evaluation metrics (return per environment evaluation)
        :param i_training_step: Training step
        :return:
        """

        jax.experimental.io_callback(self._checkpoint_base, None, update_runner, metrics, i_training_step)

    def _checkpoint_base(
            self,
            update_runner: Runner,
            metrics: Dict[str, Float[Array, "1"]],
            i_training_step: int
    ) -> None:
        """
        Implements checkpointing, to be wrapped in a Python callback. Checkpoints the following:
        - The training runner object.
        - Returns of the evaluation episodes
        The average return over the evaluated episodes is used as the checkpoint metric.
        :param update_runner: The runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param metrics: Dictionary of evaluation metrics (return per episode evaluation)
        :param i_training_step: Training step
        :return:
        """

        if self.checkpointing:

            ckpt = {
                "runner": update_runner,
                "terminated": metrics["terminated"],
                "truncated": metrics["truncated"],
                "final_rewards": metrics["final_rewards"],
                "returns": metrics["returns"]
            }

            save_args = orbax_utils.save_args_from_target(ckpt)

            self.checkpoint_manager.save(
                # Use maximum number of steps reached in previous training. Set to zero by default during agent
                # initialization if a new training is executed. In case of continuing training, the checkpoint of step
                # zero replaces the last checkpoint of the previous training. The two checkpoints are the same.
                i_training_step+self.previous_training_max_step,
                ckpt,
                save_kwargs={'save_args': save_args},
            )

    @partial(jax.jit, static_argnums=(0,))
    def _training_step(
            self,
            update_runner: Runner,
            i_training_batch: int
    ) -> Tuple[Runner, Dict[str, Float[Array, "n_agents"]]]:
        """
        Performs trainings steps to update the agent per training batch.
        :param update_runner: The runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters.
        :param i_training_batch: Training batch loop counter.
        :return: Tuple with updated runner and dictionary of metrics.
        """

        n_training_steps = self.config.n_steps - self.config.n_steps // self.config.eval_frequency * i_training_batch
        n_training_steps = jnp.clip(n_training_steps, 1, self.config.eval_frequency)

        update_runner = lax.fori_loop(0, n_training_steps, self._update_step, update_runner)

        if self.eval_during_training:
            metrics = self._generate_metrics(runner=update_runner, update_step=i_training_batch)
            i_training_step = self.config.eval_frequency * (i_training_batch + 1)
            i_training_step = jnp.minimum(i_training_step, self.config.n_steps)
            if self.checkpointing:
                self._checkpoint(update_runner, metrics, i_training_step)
        else:
            metrics = {}

        return update_runner, metrics

    @partial(jax.jit, static_argnums=(0,))
    def train(
            self,
            rng: PRNGKeyArray,
            hyperparams: HyperParameters
    ) -> Tuple[Runner, Dict[str, Float[Array, "n_agents"]]]:
        """
        Trains the agents. A jax_tqdm progressbar has been added in the lax.scan loop.
        :param rng: Random key for initialization. This is the original key for training.
        :param hyperparams: An instance of HyperParameters for training.
        :return: The final state of the step runner after training and the training metrics accumulated over all
                 training batches and steps.
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

        # Checkpoint initial state
        if self.eval_during_training:
            metrics_start = self._generate_metrics(runner=update_runner, update_step=0)
            if self.checkpointing:
                self._checkpoint(update_runner, metrics_start, self.previous_training_max_step)

        # Initialize agent updating functions, which can be avoided to be done within the training loops.
        actor_grad_fn = jax.grad(self._actor_loss, has_aux=True, allow_int=True)
        self._actor_minibatch_fn = lambda x, y: self._actor_minibatch_update(x, y, actor_grad_fn)

        critic_grad_fn = jax.grad(self._critic_loss, allow_int=True)
        self._critic_minibatch_fn = lambda x, y: self._critic_minibatch_update(x, y, critic_grad_fn)

        # Train, evaluate, checkpoint
        n_training_batches = self.config.n_steps // self.config.eval_frequency
        progressbar_desc = f'Training batch (training steps = batch x {self.config.eval_frequency})'

        runner, metrics = lax.scan(
            scan_tqdm(n_training_batches, desc=progressbar_desc)(self._training_step),
            update_runner,
            jnp.arange(n_training_batches),
            n_training_batches
        )

        if self.eval_during_training:
            metrics = {
                key: jnp.concatenate((metrics_start[key][jnp.newaxis, :], metrics[key]), axis=0)
                for key in metrics.keys()
            }
        else:
            metrics= {}

        return runner, metrics

    @abstractmethod
    def _trajectory_returns(self, value: Float[Array, "batch_size"], traj: Transition) -> Tuple[float, float]:
        """
        Calculates the returns per episode step over a batch of trajectories.
        :param value: The values of the steps in the trajectory according to the critic (including the one of the last
         state).
        :param traj: The trajectory batch.
        :return: A tuple of returns.
        """

        raise NotImplementedError

    @abstractmethod
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
    def _actor_loss(
            self,
            training: TrainState,
            obs: Float[Array, "n_rollout batch_size obs_size"],
            action: Float[Array, "n_rollout batch_size"],
            log_prob_old: Float[Array, "n_rollout batch_size"],
            advantage: ReturnsType,
            hyperparams: HyperParameters
    )-> Tuple[Float[Array, "1"], Float[Array, "1"]]:
        """
        Calculates the actor loss. For the REINFORCE agent, the advantage function is the difference between the
        discounted returns and the value as estimated by the critic.
        :param training: The actor TrainState object.
        :param obs: The obs in the trajectory batch.
        :param action: The actions in the trajectory batch.
        :param log_prob_old: Log-probabilities of the old policy collected over the trajectory batch.
        :param advantage: The advantage over the trajectory batch.
        :param hyperparams: The HyperParameters object used for training.
        :return: A tuple containing the actor loss and the KL divergence (for early checking stopping criterion).
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(
            self,
            training: TrainState,
            obs: Float[Array, "n_rollout batch_size obs_size"],
            targets: Float[Array, "batch_size n_rollout"],
            hyperparams: HyperParameters
    ) -> float:
        """
        Calculates the critic loss.
        :param training: The critic TrainState object.
        :param obs: The obs in the trajectory batch.
        :param targets: The returns over the trajectory batch, which act as the targets for training the critic.
        :param hyperparams: The HyperParameters object used for training.
        :return: The critic loss.
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition) -> Tuple[ActorLossInputType]:
        """
        Prepares the input required by the actor loss function. The input is reshaped so that it is split into
        minibatches.
        :param update_runner: The runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the actor loss function.
        """

        raise NotImplementedError

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner: Runner, traj_batch: Transition) -> CriticLossInputType:
        """
        Prepares the input required by the critic loss function. The input is reshaped so that it is split into
        minibatches.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the critic loss function.
        """

        raise NotImplementedError

    @abstractmethod
    def _entropy(self, training: TrainState, obs: ObsType)-> Float[Array, "1"]:
        raise NotImplemented

    @abstractmethod
    def _log_prob(
            self,
            training: TrainState,
            params: FrozenDict,
            obs: ObsType,
            action: ActionType
    ) -> Float[Array, "n_actors"]:
        raise NotImplemented

    @abstractmethod
    def _sample_actions(
            self,
            rng: PRNGKeyArray,
            training: TrainState,
            obs: ObsType
    ) -> ActionType:
        raise NotImplemented

    """ METHODS FOR APPLYING AGENT"""

    @abstractmethod
    def policy(self, training: TrainState, obs: ObsType) -> ActionType:
        """
        Evaluates the action of the optimal policy (argmax) according to the trained agent for the given state.
        :param obs: The current obs of the episode step in array format.
        :return:
        """
        raise NotImplemented

    def _eval_agent(
            self,
            rng: PRNGKeyArray,
            actor_training: TrainState,
            critic_training: TrainState,
            n_episodes: int = 1
    ) -> Dict[str, Float[Array, "n_agents"] | Bool[Array, "1"]]:
        """
        Evaluates the agents for n_episodes complete episodes using 'lax.while_loop'.
        :param rng: A random key used for evaluating the agent.
        :param actor_training: The actor TrainState object (either mid- or post-training).
        :param critic_training: The critic TrainState object (either mid- or post-training).
        :param n_episodes: The update_runner object used during training.
        :return: The sum of rewards collected over n_episodes episodes.
        """

        rng_eval = jax.random.split(rng, n_episodes)
        rng, obs, envstate = jax.vmap(self.env_reset)(rng_eval)

        eval_runner = (
            envstate,
            obs,
            actor_training,
            jnp.zeros(1, dtype=jnp.bool).squeeze(),
            jnp.zeros(1, dtype=jnp.bool).squeeze(),
            jnp.zeros(self.n_actors),
            jnp.zeros(self.n_actors),
            rng,
        )

        eval_runners = jax.vmap(
            lambda s, t, u, v, w, x, y, z: (s, t, u, v, w, x, y, z),
            in_axes=(0, 0, None, None, None, None, None, 0)
        )(*eval_runner)

        eval_runner = jax.vmap(lambda x: lax.while_loop(self._eval_cond, self._eval_body, x))(eval_runners)
        _, _, _, terminated, truncated, final_rewards, returns, _ = eval_runner

        return self._eval_metrics(terminated, truncated, final_rewards, returns)

    def _eval_metrics(
            self,
            terminated: Bool[Array, "1"],
            truncated: Bool[Array, "1"],
            final_rewards: Float[Array, "1"],
            returns: Float[Array, "1"]
    ) -> Dict[str, Float[Array, "1"] | Bool[Array, "1"]]:
        """
        Evaluate the metrics.
        :param terminated: Whether the episode finished by termination.
        :param truncated: Whether the episode finished by truncation.
        :param final_rewards: The rewards collected in the final step of the episode.
        :param returns: The sum of rewards collected during the episode.
        :return: Dictionary combining the input arguments and the case-specific special metrics.
        """
        metrics = {
            "terminated": terminated,
            "truncated": truncated,
            "final_rewards": final_rewards,
            "returns": returns
        }

        return metrics

    @partial(jax.jit, static_argnums=(0,))
    def _eval_body(self, eval_runner: EvalRunnerType) -> EvalRunnerType:
        """
        A step in the episode to be used with 'lax.while_loop' for evaluation of the agent in a complete episode.
        :param eval_runner: A tuple containing information about the environment state, the actor and critic training
        states, whether the episode is terminated (for checking the condition in 'lax.while_loop'), the sum of rewards
        over the episode and a random key.
        :return: The updated eval_runner tuple.
        """

        envstate, obs, actor_training, terminated, truncated, reward, returns, rng = eval_runner

        actions = self.policy(actor_training, obs)

        rng, next_obs, next_envstate, reward, done, info = self.env_step(rng, envstate, actions)

        terminated = info["terminated"]
        truncated = info["truncated"]

        returns += reward

        eval_runner = (next_envstate, next_obs, actor_training, terminated, truncated, reward, returns, rng)

        return eval_runner

    @partial(jax.jit, static_argnums=(0,))
    def _eval_cond(self, eval_runner: EvalRunnerType) -> Bool[Array, "1"]:
        """
        Checks whether the episode is terminated, meaning that the 'lax.while_loop' can stop.
        :param eval_runner: A tuple containing information about the environment state, the actor and critic training
        states, whether the episode is terminated (for checking the condition in 'lax.while_loop'), the sum of rewards
        over the episode and a random key.
        :return: Whether the episode is terminated, which means that the while loop must stop.
        """

        _, _, _, terminated, truncated, _, _, _ = eval_runner
        return jnp.logical_and(jnp.logical_not(terminated), jnp.logical_not(truncated))

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

    def log_hyperparams(self, hyperparams: HyperParameters) -> None:
        """
        Logs training hyperparameters in a text file. To be used outside training.
        :param hyperparams: An instance of HyperParameters for training.
        :return:
        """

        output_lst = [field + ': ' + str(getattr(hyperparams, field)) for field in hyperparams._fields]
        output_lst = ['Hyperparameters:'] + output_lst
        output_lst = '\n'.join(output_lst)

        if self.checkpointing:
            with open(os.path.join(self.config.checkpoint_dir, 'hyperparameters.txt'), "w") as f:
                f.write(output_lst)

    def collect_training(
            self,
            runner: Optional[Runner] = None,
            metrics: Optional[Dict[str, Float[Array, "1"]]] = None,
            previous_training_max_step: int = 0
    ) -> None:
        """
        Collects training or restored checkpoint of output (the final state of the runner after training and the
        collected metrics).
        :param runner: The runner object, containing information about the current status of the actor's/
        critic's training, the state of the environment and training hyperparameters. This is at the state reached at
        the end of training.
        :param metrics: Dictionary of evaluation metrics (return per environment evaluation)
        :param previous_training_max_step: Maximum step reached during training.
        :return:
        """

        self.agent_trained = True
        self.previous_training_max_step = previous_training_max_step
        self.training_runner = runner
        self.training_metrics = metrics
        n_evals = list(metrics.values())[0].shape[0]
        self.eval_steps_in_training = jnp.arange(n_evals) * self.config.eval_frequency
        self._pp()

    def _pp(self) -> None:
        """
        Post-processes the training results, which includes:
            - Setting the policy actor and critic TrainStates of a Runner object (e.g. last in training of restored).
        :return:
        """

        self.actor_training = self.training_runner.actor_training
        self.critic_training = self.training_runner.critic_training

    def summarize(
            self,
            metrics: Annotated[NDArray[np.float32], "size_metrics"] | Float[Array, "size_metrics"]
    ) -> MetricStats:
        """
        Summarizes collection of per-episode metrics.
        :param metrics: Metric per episode.
        :return: Summary of metric per episode.
        """

        return MetricStats(
            episode_metric=metrics,
            mean=metrics.mean(axis=-1),
            var=metrics.var(axis=-1),
            std=metrics.std(axis=-1),
            min=metrics.min(axis=-1),
            max=metrics.max(axis=-1),
            median=jnp.median(metrics, axis=-1),
            has_nans=jnp.any(jnp.isnan(metrics), axis=-1)
        )


class IPPO(IPPOBase):

    """
    IPPO clip agent using the GAE (PPO2) for calculating the advantage. The actor loss function standardizes the
    advantage.
    """

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
    def _actor_loss(
            self,
            training: TrainState,
            obs: Annotated[ObsType, "n_rollout batch_size"],
            actions: Annotated[ActionType, "batch_size"],
            log_prob_old: Float[Array, "n_rollout batch_size"],
            advantage: ReturnsType,
            hyperparams: HyperParameters
    )-> Tuple[Float[Array, "1"], Float[Array, "1"]]:
        """
        Calculates the actor loss. For the REINFORCE agent, the advantage function is the difference between the
        discounted returns and the value as estimated by the critic.
        :param training: The actor TrainState object.
        :param obs: The obs in the trajectory batch.
        :param actions: The actions in the trajectory batch.
        :param log_prob_old: Log-probabilities of the old policy collected over the trajectory batch.
        :param advantage: The GAE over the trajectory batch.
        :param hyperparams: The HyperParameters object used for training.
        :return: A tuple containing the actor loss and the KL divergence (for early checking stopping criterion).
        """

        """ Standardize GAE, greatly improves behaviour"""
        advantage = (advantage - advantage.mean(axis=0)) / (advantage.std(axis=0) + 1e-8)

        log_prob_vmap = jax.vmap(jax.vmap(self._log_prob, in_axes=(None, None, 0, 0)), in_axes=(None, None, 0, 0))
        log_prob = log_prob_vmap(training, training.params, obs, actions)
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

        entropy_vmap = jax.vmap(jax.vmap(self._entropy, in_axes=(None, 0)), in_axes=(None, 0))
        entropy = entropy_vmap(training, obs)

        total_loss_actor = loss_actor.mean() + hyperparams.ent_coeff * entropy.mean()

        """ Negative loss, because we want ascent but 'apply_gradients' applies descent """
        return -total_loss_actor, kl

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss(
            self,
            training: TrainState,
            obs: Annotated[ObsType, "n_rollout batch_size"],
            targets: ReturnsType,
            hyperparams: HyperParameters
    ) -> Float[Array, "1"]:
        """
        Calculates the critic loss.
        :param training: The critic TrainState object.
        :param obs: The obs in the trajectory batch.
        :param targets: The targets over the trajectory batch for training the critic.
        :param hyperparams: The HyperParameters object used for training.
        :return: The critic loss.
        """

        value_vmap = jax.vmap(jax.vmap(training.apply_fn, in_axes=(None, 0)), in_axes=(None, 0))
        value = value_vmap(training.params, obs)
        residuals = value - targets
        value_loss = jnp.mean(residuals ** 2)
        critic_total_loss = hyperparams.vf_coeff * value_loss

        return critic_total_loss

    @partial(jax.jit, static_argnums=(0,))
    def _actor_loss_input(self, update_runner: Runner, traj_batch: Transition) -> ActorLossInputType:
        """
        Prepares the input required by the actor loss function. For the PPO agent, this entails the:
        - the actions collected over the trajectory batch.
        - the log-probability of the actions collected over the trajectory batch.
        - the returns over the trajectory batch.
        - the values over the trajectory batch as evaluated by the critic.
        - the training hyperparameters.
        The input is reshaped so that it is split into minibatches.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the actor loss function.
        """

        # Shuffle the trajectory batch to collect minibatches.
        # Poor practice in using the random key, which however doesn't influence the training, since all trajectories in
        # the batch are used per epoch.
        minibatch_idx = jax.random.choice(
            jax.random.PRNGKey(1),
            jnp.arange(self.config.batch_size),
            replace=False,
            shape=(self.config.batch_size,)
        )

        traj_minibatch = jax.tree_map(lambda x: jnp.take(x, minibatch_idx, axis=0), traj_batch)
        traj_minibatch = jax.tree_map(lambda x: x.reshape(-1, self.config.minibatch_size, *x.shape[1:]), traj_minibatch)

        return (
            traj_minibatch.obs,
            traj_minibatch.action,
            traj_minibatch.log_prob,
            traj_minibatch.advantage,
            update_runner.hyperparams
        )

    @partial(jax.jit, static_argnums=(0,))
    def _critic_loss_input(self, update_runner: Runner, traj_batch: Transition) -> CriticLossInputType:
        """
        Prepares the input required by the critic loss function. For the PPO agent, this entails the:
        - the states collected over the trajectory batch.
        - the targets (returns = GAE + next_value) over the trajectory batch.
        - the training hyperparameters.
        The input is reshaped so that it is split into minibatches.
        :param update_runner: The Runner object used in training.
        :param traj_batch: The batch of trajectories.
        :return: A tuple of input to the critic loss function.
        """

        # Shuffle the trajectory batch to collect minibatches.
        # Poor practice in using the random key, which however doesn't influence the training, since all trajectories in
        # the batch are used per epoch.
        minibatch_idx = jax.random.choice(
            jax.random.PRNGKey(1),
            jnp.arange(self.config.batch_size),
            replace=False,
            shape=(self.config.batch_size,)
        )

        traj_minibatch = jax.tree_map(lambda x: jnp.take(x, minibatch_idx, axis=0), traj_batch)
        traj_minibatch = jax.tree_map(lambda x: x.reshape(-1, self.config.minibatch_size, *x.shape[1:]), traj_minibatch)

        return (
            traj_minibatch.obs,
            traj_minibatch.advantage + traj_minibatch.value,
            update_runner.hyperparams
        )


if __name__ == "__main__":
    pass

