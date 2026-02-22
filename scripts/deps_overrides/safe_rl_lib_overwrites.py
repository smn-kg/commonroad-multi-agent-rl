#print("[PATCH] CommonRoadEnv.simulation_preprocessing overridden")
from safe_rl_lib.data_models.model_configs import BaseTorchModelConfig, ConvArchitecture, LinearArchitecture
from safe_rl_lib.data_models.policy_configs import SACConfig, PPOConfig, A2CConfig
from safe_rl_lib.data_models.trainer_configs import OffPolicyTrainerConfig, OnPolicyTrainerConfig
from safe_rl_lib.tianshou_utils.nets import create_critic, create_base_net, ConvNet
from tianshou.utils.net.common import ActorCritic

from safe_rl_lib.tianshou_utils import nets
import torch
from copy import copy
import gymnasium as gym
import torch.nn as nn
import tianshou as ts

from tianshou.utils.net.continuous import Actor, ActorProb, Critic
from tianshou.utils.net.common import Net

#changed: injecting custom hyperparameters into the policy
def patched_create_policy(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, env: gym.Env
) -> ts.policy.BasePolicy:
    assert isinstance(model, ActorCritic)

    if isinstance(self.policy_config, PPOConfig):
        if self.train_config is not None:
            assert isinstance(self.train_config, OnPolicyTrainerConfig)

        cfg = self.policy_config.model_dump()

        policy = ts.policy.PPOPolicy(
            **cfg,  # CC hyperparameters from the PPOConfig class
            actor=model.actor,
            critic=model.critic,
            optim=optimizer,
            dist_fn=self.policy_distribution_fn,
            action_space=env.action_space,
        )
        return policy

    elif isinstance(self.policy_config, A2CConfig):
        if self.train_config is not None:
            assert isinstance(self.train_config, OnPolicyTrainerConfig)
        policy = ts.policy.A2CPolicy(
            **self.policy_config.model_dump(),
            actor=model.actor,
            critic=model.critic,
            optim=optimizer,
            dist_fn=self.policy_distribution_fn,
            action_space=env.action_space,
        )

    elif isinstance(self.policy_config, SACConfig):
        if self.train_config is not None:
            assert isinstance(self.train_config, OffPolicyTrainerConfig)
        critic2 = create_critic(env, self.model_config, use_action_shape=True, concat=True)
        critic2_optimizer = self._create_optimizer(critic2)

        actor_optimizer = self._create_optimizer(model.actor)
        critic_optimizer = self._create_optimizer(model.critic)

        policy = ts.policy.SACPolicy(
            **self.policy_config.model_dump(),
            actor=model.actor,
            actor_optim=actor_optimizer,
            critic=model.critic,
            critic_optim=critic_optimizer,
            critic2=critic2,
            critic2_optim=critic2_optimizer,
            action_space=env.action_space,
        )
    else:
        raise NotImplementedError(f'Setup not implemented for {self.policy_config}')

    return policy

#changed: critic shape changed to take global observations
def _patched_create_actor_critic(
    env: gym.Env,
    actor_config: BaseTorchModelConfig,
    critic_config: BaseTorchModelConfig | None = None,
    probabilistic_actor: bool = False,
    use_critic_action_shape: bool = False,
    concat_net: bool = False,

):

    if isinstance(env.action_space, gym.spaces.Box):
        action_shape = env.action_space.shape
    else:
        raise NotImplementedError

    if critic_config is None:
        critic_config = copy(actor_config)

    actor_net = create_base_net(env, actor_config)

    if probabilistic_actor:
        actor = ActorProb(actor_net, action_shape, unbounded=True, device=actor_config.device).to(
            actor_config.device
        )

    else:
        actor = Actor(actor_net, action_shape, device=actor_config.device).to(actor_config.device)

    critic_net = _patched_create_base_net(
        env, critic_config, use_action_shape=use_critic_action_shape, concat=concat_net,  state_shape_override=(96,) #32 * num_agents
    )
    critic = Critic(critic_net, device=critic_config.device).to(critic_config.device)

    # CC initial critic bias for faster convergence
    with torch.no_grad():
        critic.last.model[-1].bias.fill_(-9.0)

    actor_critic = ActorCritic(actor, critic)

    return actor_critic


def _patched_create_base_net(
    env: gym.Env,
    model_config: BaseTorchModelConfig,
    use_action_shape: bool = False,
    concat: bool = False,
    state_shape_override=None,                                                                  #CCt modified
) -> nn.Module:
    if isinstance(env.observation_space, gym.spaces.Box):
        state_shape = state_shape_override or env.observation_space.shape                       #CCt modified
    else:
        raise NotImplementedError

    if model_config.activation_fn_name == 'ReLU':
        activation = nn.ReLU
    elif model_config.activation_fn_name == 'LeayReLU':
        activation = nn.LeakyReLU
    else:
        raise NotImplementedError

    action_shape = 0
    if use_action_shape:
        if isinstance(env.action_space, gym.spaces.Box):
            action_shape = env.action_space.shape
        else:
            raise NotImplementedError

    if isinstance(model_config.architecture, ConvArchitecture):
        # noinspection PyTypeChecker
        architecture: ConvArchitecture = model_config.architecture
        base_net = ConvNet(
            c=state_shape[2],
            h=state_shape[0],
            w=state_shape[1],
            architecture=architecture,
            activation=activation,
            device=model_config.device,
        )
    elif isinstance(model_config.architecture, LinearArchitecture):
        # noinspection PyTypeChecker
        architecture: LinearArchitecture = model_config.architecture
        base_net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=architecture.hidden_sizes,
            device=model_config.device,
            activation=activation,
            concat=concat,
            norm_layer=nn.LayerNorm,
        )
    else:
        raise NotImplementedError

    return base_net