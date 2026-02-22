from safe_rl_envs.envs.commonroad.commonroad_env import CommonRoadEnv
from deps_overrides.commonroad_env_overrides import (
    _patched_step,
    _patched_simulation_preprocessing,
)

import pettingzoo.utils.conversions as conv
import pettingzoo.utils.env as pz_env
from deps_overrides.pettingzoo_overrides import (
    patched_parallel_to_aec_wrapper,
    _patched_was_dead_step,
    _patched_deads_step_first,
)

from tianshou.data.collector import Collector
from deps_overrides.collector_override import (
    _patched_collect,
    _patched_compute_action_policy_hidden,
)

import tianshou.policy.modelfree.ppo as ppo_mod
import tianshou.policy.modelfree.a2c as a2c_mod
from tianshou.policy.multiagent.mapolicy import MultiAgentPolicyManager

from deps_overrides.policy_overrides import (
    _patched_ppo_learn,
    _patched_a2c_compute_returns,
    _patched_mapolicy_process_fn,
)

from safe_rl_lib.setup_pipelines.stochastic_pg import StochasticPGSetup
import safe_rl_lib.setup_pipelines.stochastic_pg as spg
import safe_rl_lib.setup_pipelines.deterministic_pg as dpg
import safe_rl_lib.tianshou_utils.nets as nets

from deps_overrides.safe_rl_lib_overwrites import (
    patched_create_policy,
    _patched_create_actor_critic,
    _patched_create_base_net
)


def apply_all_overrides() -> None:
    """
    Apply all dependency overrides (monkey patches) used by this repo.

    Call this once at the very top of your entry script (train/eval/render)
    before importing/instantiating envs, policies, collectors, or trainers.
    """

    # =========================
    # CommonRoad Env overrides
    # =========================

    CommonRoadEnv.step = _patched_step
    CommonRoadEnv.simulation_preprocessing = _patched_simulation_preprocessing

    from safe_rl_lib.setup_pipelines.base import SetupPipeline as LibSetupPipeline
    from deps_overrides.setup_pipelines_base_override import SetupPipeline as MySetupPipeline

    LibSetupPipeline._env_function = MySetupPipeline._env_function
    LibSetupPipeline._env_function_test = MySetupPipeline._env_function_test
    LibSetupPipeline._create_environments = MySetupPipeline._create_environments
    LibSetupPipeline._create_collector = MySetupPipeline._create_collector
    LibSetupPipeline.create_trainer = MySetupPipeline.create_trainer
    LibSetupPipeline.create_single_env = MySetupPipeline.create_single_env
    LibSetupPipeline.create_policy_for_eval = MySetupPipeline.create_policy_for_eval

    # =========================
    # PettingZoo overrides
    # =========================
    # parallel -> aec wrapper override
    conv.parallel_to_aec_wrapper = patched_parallel_to_aec_wrapper

    # dead-agent handling overrides
    AECEnv = pz_env.AECEnv
    AECEnv._was_dead_step = _patched_was_dead_step
    AECEnv._deads_step_first = _patched_deads_step_first

    # =========================
    # Tianshou Collector overrides
    # =========================
    Collector._collect = _patched_collect
    Collector._compute_action_policy_hidden = _patched_compute_action_policy_hidden

    # =========================
    # Tianshou Policy overrides
    # =========================
    ppo_mod.PPOPolicy.learn = _patched_ppo_learn
    a2c_mod.A2CPolicy._compute_returns = _patched_a2c_compute_returns
    MultiAgentPolicyManager.process_fn = _patched_mapolicy_process_fn

    # =========================
    # safe_rl_lib overwrites
    # =========================
    StochasticPGSetup._create_policy = patched_create_policy

    # patch create_actor_critic where those setup modules reference it
    nets.create_actor_critic = _patched_create_actor_critic
    spg.create_actor_critic = _patched_create_actor_critic
    dpg.create_actor_critic = _patched_create_actor_critic

    nets.create_base_net = _patched_create_base_net
    spg.create_base_net = _patched_create_base_net
    dpg.create_base_net = _patched_create_base_net























