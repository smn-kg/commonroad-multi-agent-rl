from dep_overrides import apply_all_overrides
apply_all_overrides()

import warnings
import logging
logging.getLogger("tianshou").setLevel(logging.CRITICAL)

import os
import torch
import argparse
import datetime
import pprint
import sys
import warnings
from pathlib import Path
import numpy as np
import gymnasium as gym
import tianshou as ts
import wandb
import matplotlib
import os, random, torch
import warnings

from commonroad.common.file_reader import CommonRoadFileReader
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy.base import BasePolicy
from tianshou.utils import TensorboardLogger, WandbLogger
from commonroad.common.solution import VehicleModel
from gymnasium.envs.registration import register
from multiagent_env import MultiAgentEnvConfig, MultiAgentCommonRoadEnv
from parallel_env import CRPettingZoo
from types import ModuleType


from safe_rl_envs.configs.env_configs import CommonRoadEnvConfig
from safe_rl_envs.configs.commonroad_config import *
from safe_rl_envs.configs.commonroad_safe_action_set_config import SafeActionSetConfig
from safe_rl_envs.envs.commonroad.wrappers.record_commonroad_statistics import RecordCommonroadStatistics
from safe_rl_envs.envs.commonroad.wrappers.render_commonroad import RenderCommonroad
from safe_rl_envs.envs.commonroad.simulation.simulation import Simulation

from safe_rl_lib.data_models.model_configs import StochasticPGModelConfig, LinearArchitecture
from safe_rl_lib.data_models.policy_configs import PPOConfig
from safe_rl_lib.data_models.trainer_configs import OnPolicyTrainerConfig, OffPolicyTrainerConfig
from safe_rl_lib.data_models.utils import save_config
from safe_rl_lib.setup_pipelines.stochastic_pg import StochasticPGSetup

# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings(
    "ignore",
    message=r"ep_return should be a scalar *",
)

os.environ["WANDB_MODE"] = "dryrun"


'''
PPO scripts entrypoint for the multi-agent CommonRoad environment (CTDE).

Parses CLI args, builds env/policy/trainer, runs scripts/evaluation, and
(optionally) exports episode trajectories to CommonRoad XML for video rendering.
'''


seed =   3785 #18 #9 #8
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed); os.environ["PYTHONHASHSEED"]=str(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

BASE_DIR = Path(__file__).resolve().parent
#MODULE FROM FILE LOADER
def load_module_from_file(file_path: Path) -> ModuleType:
    # Create a fresh module
    module = ModuleType(file_path.stem)
    # Read and execute its source into the module’s namespace
    code = file_path.read_text()
    exec(compile(code, str(file_path), "exec"), module.__dict__)
    return module

temp_norm = load_module_from_file(BASE_DIR / "obs_stats.py")
GLOBAL_OBS_STAT = temp_norm.GLOBAL_OBS_STAT


register(
    id="MultiAgentCommonRoadEnv-v0",
    entry_point="multiagent_env:MultiAgentCommonRoadEnv",
)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_dir", type=str, default="scenarios")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)#4
    # parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=3)

    parser.add_argument("--buffer_size", type=int, default=25600*1.5)
    parser.add_argument("--steps_per_epoch", type=int, default=25600*1.5)
    parser.add_argument("--steps_per_collect", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--update_per_step", type=float, default=0.1)

    parser.add_argument("--training_num", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=1)#5
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--normalize_obs", default=False, action="store_true")
    parser.add_argument("--logdir", type=Path, default=BASE_DIR / "logs")
    parser.add_argument("--cachedir", type=str, default="cache")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume_path", type=Path, default=None)
    parser.add_argument("--resume_id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb_project", type=str, default="safe-rl-autodrive")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    parser.add_argument("--save_buffer_name", type=str, default=None)


    parser.add_argument("--reward_velocity_towards_goal_lat", type=float, default=0.5)
    parser.add_argument("--reward_velocity_towards_goal_long", type=float, default=0.5)

    # remove dense penalties
    parser.add_argument("--reward_lat_offset_lane_center", type=float, default=-0.0)
    parser.add_argument("--reward_lat_offset_ref_path", type=float, default=0.0)

    parser.add_argument("--reward_jerk_lat", type=float, default=0.0)
    parser.add_argument("--reward_jerk_long", type=float, default=0.0)

    parser.add_argument("--reward_control_effort_long", type=float, default=0.0)
    parser.add_argument("--reward_control_effort_lat", type=float, default=0.0)

    parser.add_argument("--reward_max_s_position", type=float, default=0.0)
    parser.add_argument("--reward_friction_violation", type=float, default=0.0)

    # strong terminal structure
    parser.add_argument("--reward_goal_reached", type=int, default=500)
    parser.add_argument("--reward_collision", type=int, default=-350)
    parser.add_argument("--reward_off_road", type=int, default=-500)
    parser.add_argument("--reward_time_out", type=int, default=-300)
    parser.add_argument("--reward_exception", type=float, default=-300)



    parser.add_argument(
        "--safe_action_mode",
        type=str,
        default="none",
        choices=["none", "ray", "replace_sample", "projection", "distributional"],
    )
    parser.add_argument("--prediction_horizon", type=int, default=12)
    parser.add_argument("--anticipated_ego_a_lon_max", type=float, default=5.0)
    parser.add_argument("--anticipated_ego_a_lat_max", type=float, default=5.0)
    parser.add_argument("--anticipated_num_steps_k_to_i", type=int, default=2)
    parser.add_argument("--observe_action_set", default=False, action="store_true")
    parser.add_argument("--enforce_goal_reaching", default=False, action="store_true")
    parser.add_argument("--distributional_entropy_simplification", default=False, action="store_true")

    
    return parser.parse_args()


def main(args: argparse.Namespace = get_args()) -> None:

    warnings.filterwarnings("ignore", category=DeprecationWarning)  # Suppress DeprecationWarnings

    args.task = "commonroad"
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    scenario_dir = PROJECT_ROOT / "scenarios"
    print("scenario_dir", scenario_dir)

    if not scenario_dir.is_dir():
        raise RuntimeError(f"{scenario_dir!r} is not a directory")

    scenario_paths = [p for p in scenario_dir.iterdir() if p.suffix == ".xml"]

    #CC log path
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    if args.safe_action_mode == "none":
        args.algo_name = "ppo"
    else:
        args.algo_name = f"ppo_{args.safe_action_mode}"

    args.log_name = Path(args.task, args.algo_name, "latest").__str__()
    log_path = Path(args.logdir, args.log_name)
    log_path.mkdir(parents=True, exist_ok=True)

    agent_config = AgentConfig(
        vehicle_model = VehicleModel.PM,
        align_pm_action_to="lane",
        rescale_pm_action_to_friction_circle=False,
    )

    ego_observation_config = EgoObservationConfig(
        observe_v_ego=True,
        observe_position=False,
        observe_orientation=False,
        observe_a_ego=True,
        observe_jerk_lat_ego=True,
        observe_jerk_long_ego=True,
        observe_steering_angle=False,
        observe_yaw_rate=True,
        observe_is_friction_violation=True,
    )

    lanelet_observation_config = LaneletObservationConfig(
        strict_offroad_check=False,
        observe_left_marker_distance=True,
        observe_right_marker_distance=True,
        observe_left_road_edge_distance=True,
        observe_right_road_edge_distance=True,
        observe_lat_centerline_offset=True,
        observe_curvature_lane=True,
        observe_orientation_error_lane=True,
        observe_is_offroad=False,
    )

    goal_observation_config = GoalObservationConfig(
        relax_is_goal_reached=True,
        observe_distance_goal_long=True,
        observe_distance_goal_lat=True,
        observe_velocity_towards_goal_long=True,
        observe_velocity_towards_goal_lat=True,
        observe_lat_offset_ref_path=True,
        observe_orientation_error_refpath=True,
        observe_curvature_ref_path=True,
        observe_is_goal_reached=False,
        observe_is_timeout=False,
    )

    surrounding_observation_config = SurroundingObservationConfig(
        observe_lane_rect_p_rel=True,
        observe_lane_rect_v_rel=True,
        observe_is_collision=False,
        max_observation_dist=70.0,
    )

    dense_reward_config = DenseRewardConfig(
        reward_velocity_towards_goal_lat=args.reward_velocity_towards_goal_lat,
        reward_velocity_towards_goal_long=args.reward_velocity_towards_goal_long,
        reward_lat_offset_lane_center=args.reward_lat_offset_lane_center,
        reward_orienation_error_lane = 0.0,
        reward_lat_offset_ref_path=args.reward_lat_offset_ref_path,
        reward_jerk_lat=args.reward_jerk_lat,
        reward_jerk_long=args.reward_jerk_long,
        reward_control_effort_long=args.reward_control_effort_long,
        reward_control_effort_lat=args.reward_control_effort_lat,
    )

    sparse_reward_config = SparseRewardConfig(
        reward_goal_reached=args.reward_goal_reached,
        reward_goal_reached_faster=args.reward_goal_reached,
        reward_goal_reached_out_of_time=args.reward_goal_reached,
        reward_collision=args.reward_collision,
        reward_off_road=args.reward_off_road,
        reward_time_out=args.reward_time_out,
        reward_max_s_position=args.reward_max_s_position,
        reward_friction_violation=args.reward_friction_violation,
        reward_exception = args.reward_exception,
    )

    termination_config = TerminationConfig(
        terminate_on_collision=True
    )

    render_config = RenderConfig(
        zoom_in_on_ego=False,
        zoom_in_on_planning_problem=False,
        fig_size=(15,5),
    )


    env_config = MultiAgentEnvConfig(
        scenario_paths= scenario_paths,
        pick_random_scenario=True,
        render_mode='rgb_array',
        pickle_scenarios=True,
        agent_config=agent_config,
        ego_observation_config=ego_observation_config,
        lanelet_observation_config=lanelet_observation_config,
        goal_observation_config=goal_observation_config,
        surrounding_observation_config=surrounding_observation_config,
        dense_reward_config=dense_reward_config,
        sparse_reward_config=sparse_reward_config,
        termination_config=termination_config,
        render_config=render_config,
        logging_path=log_path,
        log_timing=True,
        log_timing_to_csv=True,
    )

    train_config = OnPolicyTrainerConfig(
        n_train_envs=args.training_num,
        n_test_envs=args.test_num,
        steps_per_epoch=args.steps_per_epoch,
        steps_per_collect=args.steps_per_collect,
        buffer_size=args.buffer_size,
        episodes_per_test=args.test_num,
        n_epochs=args.epoch,
    )

    model_config = StochasticPGModelConfig(
        lr=args.lr, batch_size=args.batch_size, architecture=LinearArchitecture(hidden_sizes=[args.hidden_size] * args.num_hidden_layers)
    )

    policy_config = PPOConfig()

    #CC Save all the configs
    save_config(policy_config, log_path.joinpath('ppo_config.yaml'))
    save_config(train_config, log_path.joinpath('train_config.yaml'))
    save_config(model_config, log_path.joinpath('model_config.yaml'))
    save_config(env_config, log_path.joinpath('env_config.yaml'))


    def env_wrapper_function(env):
        env = RecordCommonroadStatistics(
            env, 
            buffer_length=250, 
            special_termination_reasons=["no_safe_action", "empty_relevant_action_set", "exception_in_safe_action_wrapper", "initial_condition_invalid", "no_driving_corridor", "velocity_violation"], #corr typo here in vel vio
            log_fn=lambda termination_dict: logger.write("test/termination", trainer.env_step, termination_dict)
            )

        return env
    
    def test_env_wrapper_function(env):   #merged with train wrapper for now might change later
        env = RenderCommonroad(env, episode_trigger=lambda x: True, log_fn=lambda img: writer.add_image("test/render", img, global_step=trainer.env_step, dataformats="HWC"))
        return env_wrapper_function(env)


    writer = SummaryWriter(str(log_path),
                           max_queue=2000, flush_secs=15)

    logger = TensorboardLogger(writer)

    experiment_setup = StochasticPGSetup(
        policy_config=policy_config,
        model_config=model_config,
        train_config=train_config,
        env_config=env_config,
        env_wrapper_function=env_wrapper_function,
        test_env_wrapper_function=test_env_wrapper_function,
        logger=logger,
        # n_agents=1,
    )

    trainer = experiment_setup.create_trainer(
        logdir=str(log_path),
        writer=writer,
        logger=logger,
    )


    def watch() -> None:
        print("Setup test envs ...")
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            test_envs = ts.env.DummyVectorEnv(
                [experiment_setup.create_single_env for _ in range(args.test_num)]
            )
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=False,
                save_only_last_obs=False,
            )
            collector = Collector(trainer.policy, test_envs, buffer, exploration_noise=False)
            collector.reset()
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")

            best_path = Path(__file__).resolve().parent / "outputs" / "checkpoints" / "best_policy.pth"
            ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
            trainer.policy.load_state_dict(ckpt["model"])

            if ckpt.get("norm_obs"):
                s = ckpt["norm_obs"]
                GLOBAL_OBS_STAT.mean[:] = s["mean"]
                GLOBAL_OBS_STAT.M2[:] = s["M2"]
                GLOBAL_OBS_STAT.n = s["n"]


            while True:
                trainer.test_collector.reset()
                result = trainer.test_collector.collect(n_episode=1, render=False)

                env0 = trainer.test_collector.env.workers[0].env
                while True:
                    if hasattr(env0, "env"):
                        env0 = env0.env
                        continue
                    if hasattr(env0, "_env"):
                        env0 = env0._env
                        continue
                    break

                term = getattr(env0, "_prev_term_reason", {})
                print("prev termination reasons:", term)

                allowed = {
                    "max_s_position",
                    "is_goal_reached_success",
                    "is_goal_reached_faster",
                }

                success = (
                        len(term) == len(env0.possible_agents) and
                        all(r in allowed for r in term.values())
                )

                if success:
                    print("All agents have reached the  goal.")
                    break
                else:
                    print("Episode rejected.")

            env0.export_episode_xml("outputs/simulated_scenarios/run_0001.xml")

        result.pprint_asdict()

    if args.watch:
        watch()
        sys.exit(0)

    import time

    print(f"[RESUME] starting from env_step={trainer.env_step}, "
          f"epoch={trainer.epoch}, max_epoch={trainer.max_epoch}")

    trainer.reset()
    #CC actual scripts run command
    start_time = time.time()
    result = trainer.run()
    elapsed = time.time() - start_time


    # format nicely
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)

    pprint.pprint(result)

    writer.flush()
    writer.close()

    watch()
    print(f"[RUNTIME] total scripts took {int(h)}h {int(m):02d}m {s:04.1f}s")

if __name__ == "__main__":
    main(get_args())
