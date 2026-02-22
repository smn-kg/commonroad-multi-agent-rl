from typing import Any
import numpy as np
import random

from commonroad.scenario.state import InitialState
from gymnasium.spaces import Dict
from gymnasium.spaces.utils import flatten
from gymnasium.spaces.utils import flatten_space
from commonroad.common.file_reader import CommonRoadFileReader
from safe_rl_envs.configs.env_configs import CommonRoadEnvConfig
from safe_rl_envs.envs.commonroad.commonroad_env import CommonRoadEnv
from safe_rl_envs.envs.commonroad.simulation.simulation import Simulation
from safe_rl_envs.envs.commonroad.simulation.agent import Agent
from pathlib import Path

agents: list[Agent]


class MultiAgentEnvConfig(CommonRoadEnvConfig):

    env_name: str = "MultiAgentCommonRoadEnv-v0"

    class Config:
        extra = "ignore"


class MultiAgentCommonRoadEnv(CommonRoadEnv):

    def __init__(self, *args, **kwargs):

        super().__init__(*args,  **kwargs)
        self.agent_order = None
        self.sim_agents = None
        self.sa_obs_space = self.observation_space
        flat = flatten_space(self.observation_space)
        single_action_space = self.action_space

        #CC get number of planning problems from scenario
        _, pp_set = CommonRoadFileReader(self.scenario_paths[0]).open()
        N = len(pp_set.planning_problem_dict)

        self._last_term_reason = {}
        self._prev_term_reason = {}

        self._prev_traj = {}


        self.observation_space = Dict({str(i): flat for i in range(N)})
        self.action_space = Dict({str(i): single_action_space for i in range(N)})

        self.agent_ids: list = []

        self.agent_ids = [str(i) for i in range(N)]
        print("number of agents: ", self.agent_ids)

        self.agent_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}

        #######DEBUG CC############
        # ===== FULL MA OBS DEBUG =====
        if not hasattr(self, "_dbg_full"):
            self._dbg_full = 0
        ####################################

    def reset(self, *, seed: int | None = None, options: dict[str,Any] | None = None):
        #CC use CommonRoadEnv reset
        raw_feats, info = super().reset(seed=seed, options=options)

        num_agents = len(getattr(self.simulation, "agents", []))


        #CC keep existing id mappings
        self.agent_ids = [str(i) for i in range(num_agents)]
        self.sim_agents = getattr(self.simulation, "agents", [])
        self.id2agent = {aid: self.sim_agents[int(aid)] for aid in self.agent_ids}
        self.live_ids = list(self.id2agent.keys())
        obs_dict, info_dict= {},{}

        if not hasattr(self, "_traj"):
            self._traj = {aid: [] for aid in self.id2agent}

        for aid, ag in self.id2agent.items():
            obs = self.oc.observe(ag)  # per-agent initial obs
            vec = flatten(self.sa_obs_space, obs).astype(np.float32, copy=False)
            obs_dict[aid] = vec

            info_dict[aid] = {
                "chosen_action": None,
                "termination_reason": None,
                "observation_dict": obs,
                "reward_dict": None,
            }

        # CENTRAL CRITIC: define fixed order + concat
        self.agent_order = list(self.agent_ids)
        obs_global = np.concatenate([obs_dict[aid] for aid in self.agent_order], axis=0)

        for aid in self.agent_order:
            info_dict[aid]["obs_global"] = obs_global


        self.last_obs = {}

        for aid, o in obs_dict.items():
            self.last_obs[aid] = o.copy()

        if getattr(self, "_last_term_reason", None):
            self._prev_term_reason = self._last_term_reason.copy()
        #self._prev_term_reason = getattr(self, "_last_term_reason", {}).copy()

        # for xml path recording
        self._last_term_reason = {}

        if getattr(self, "_traj", None) and any(len(tr) > 0 for tr in self._traj.values()):
            self._prev_traj = {aid: list(tr) for aid, tr in self._traj.items()}


        self._traj = {aid: [] for aid in self.agent_ids}

        return obs_dict, info_dict

    def step(self, action_dict: dict[str, np.ndarray]):
        self.profiler.start("step_env_total")


        try:
            #CC apply actions ONLY to live agents
            for aid in list(self.live_ids):
                ag = self.id2agent[aid]  # must exist for live agent
                if aid not in action_dict:
                    raise RuntimeError(f"[MA.step] missing action for live agent {aid}")

                ag.planner_interface.set_action(action_dict[aid])


            #CC advance the simulation one step
            self.profiler.start("step_simulation")
            self.status = self.simulation.step_sequential_simulation()
            self.profiler.stop("step_simulation")

            #CC build dict outputs keyed by PettingZoo ids
            obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict = {}, {}, {}, {}, {}

            for aid in list(self.live_ids):
                ag = self.id2agent[aid]
                if ag is None:
                    print("no corresponding agent found in ma_env!!!")
                    #CC agent missing in sim → set defaults
                    terminated_dict[aid] = True
                    truncated_dict[aid] = False
                    rew_dict[aid] = 0.0
                    obs_dict[aid] = np.zeros_like(next(iter(self.observation_space.spaces.values())).sample())
                    info_dict[aid] = {"termination_reason": "no_sim_agent"}
                    continue


                t_sim = self.simulation.global_timestep
                st = ag.current_state


                # logic for logging vehicle states
                self._traj[aid].append(st)


                obs = self.oc.observe(ag)
                vec_obs = flatten(self.sa_obs_space, obs).astype(np.float32, copy=False)
                obs_dict[aid] = vec_obs

                self.last_obs[aid] = vec_obs.copy()


                r, r_detail = self.reward_function.calc_reward(agent=ag, observation_collector=self.oc)
                rew_dict[aid] = float(r)

                done, reason, term_info = self.termination.is_terminated(agent=ag, observation_collector=self.oc)
                if not self.status and not done:
                    done = True
                    reason = ["simulation terminated"]
                terminated_dict[aid] = bool(done)
                truncated_dict[aid] = False  # TODO: set per-agent truncation if you have it

                #CC create info dict
                info_dict[aid] = {
                    "chosen_action": action_dict.get(aid, None),
                    "current_ego_state": ag.current_state,
                    "current_episode_time_step": self.simulation.global_timestep,
                    "max_episode_time_steps": ag.agent_state.goal_checker.last_goal_timestep,
                    "termination_info": term_info,
                    "termination_reason": ", ".join(reason) if reason else None,
                    "observation_dict": obs,
                    "reward_dict": r_detail,
                }

        except Exception:
            import traceback
            print("[MA.step] EXCEPTION in env; padding outputs & terminating all agents")
            print(traceback.format_exc())
            # pad with safe defaults for ALL ids so parallel API stays consistent
            obs_sample = next(iter(self.observation_space.spaces.values())).sample()
            zero_obs = np.zeros_like(obs_sample)
            obs_dict = {aid: zero_obs for aid in self.agent_ids}
            rew_dict = {aid: 0.0 for aid in self.agent_ids}
            terminated_dict = {aid: True for aid in self.agent_ids}
            truncated_dict = {aid: False for aid in self.agent_ids}
            info_dict = {aid: {} for aid in self.agent_ids}
            self.status = False


        #CENTRAL CRITIC PART
        order = self.agent_ids  # fixed ordering

        # padding template with correct shape
        z = np.zeros_like(next(iter(obs_dict.values()))) if obs_dict else None

        obs_global = np.concatenate(
            [obs_dict.get(aid, z) for aid in order],
            axis=0
        )

        for aid in info_dict:  # attach same object reference
            info_dict[aid]["obs_global"] = obs_global

        self.current_timestep += 1
        self.profiler.stop("step_env_total")

        self.live_ids = [aid for aid in self.live_ids if not terminated_dict[aid]]


        for aid, inf in info_dict.items():
            r = inf.get("termination_reason")
            if r is not None:
                self._last_term_reason[aid] = r

        return obs_dict, rew_dict, terminated_dict, truncated_dict, info_dict

    def export_episode_xml(self, out_xml):
        import copy
        from commonroad.common.file_writer import CommonRoadFileWriter
        from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType, StaticObstacle
        from commonroad.scenario.trajectory import Trajectory
        from commonroad.prediction.prediction import TrajectoryPrediction
        from commonroad.geometry.shape import Rectangle

        out_xml = Path(out_xml)
        out_xml.parent.mkdir(parents=True, exist_ok=True)

        scenario_out = copy.deepcopy(self.simulation.scenario)


        #################
        static_obs = next(iter(scenario_out.static_obstacles))
        init = static_obs.initial_state
        shape_obs = static_obs.obstacle_shape
        scenario_out.remove_obstacle(static_obs)

        T_END = 80  # must cover your rendered frames

        states = [
            InitialState(
                time_step=t,
                position=init.position,
                orientation=init.orientation,
                velocity=0.0,
            )
            for t in range(0, T_END + 1)
        ]

        traj = Trajectory(0, states)
        pred = TrajectoryPrediction(traj, shape_obs)

        fake_static = DynamicObstacle(
            obstacle_id=43,
            obstacle_type=ObstacleType.TRUCK,
            obstacle_shape=shape_obs,
            initial_state=states[0],
            prediction=pred,
        )

        scenario_out.add_objects(fake_static)
        ###############

        shape = Rectangle(length=4.5, width=2.0)
        next_id = 10_000

        traj_src = getattr(self, "_prev_traj", self._traj)

        for aid, states in traj_src.items():
            if not states:
                continue
            states = sorted(states, key=lambda s: s.time_step)

            traj = Trajectory(states[0].time_step, states)

            pred = TrajectoryPrediction(traj, shape)

            initial_state = InitialState(
                time_step=states[0].time_step,
                position=states[0].position,
                orientation=states[0].orientation,
                velocity=states[0].velocity,
            )

            otype = ObstacleType.CAR if int(aid) % 2 == 0 else ObstacleType.TRUCK

            dyn = DynamicObstacle(
                obstacle_id=next_id,
                obstacle_type=ObstacleType.CAR,
                obstacle_shape=shape,
                initial_state=initial_state,
                prediction=pred,
            )
            scenario_out.add_objects(dyn)
            next_id += 1

            # --- goal marker ---
            pp_dict = self.simulation.planning_problem_set.planning_problem_dict
            pp_keys = list(pp_dict.keys())

            pp_id = pp_keys[int(aid)]
            pp = pp_dict[pp_id]

            goal_state = pp.goal.state_list[0]  # might be a GoalRegion state

            goal_center = goal_state.position.center  # typical for Rectangle/Polygon goal
            goal_marker = StaticObstacle(
                obstacle_id=next_id,
                obstacle_type=otype,  # match color/style
                obstacle_shape=Rectangle(4.5, 2.0),
                initial_state=InitialState(time_step=0, position=goal_center, orientation=0.0, velocity=0.0),
            )
            scenario_out.add_objects(goal_marker)
            next_id += 1


        CommonRoadFileWriter(
            scenario_out,
            self.simulation.planning_problem_set,
            author="eval-export",
            source="RL rollout",
        ).write_to_file(str(out_xml))


def _patched_simulation_preprocessing(self) -> None:

    #CC simulation override
    if self.pick_random_scenario:
        self.scenario_index = random.randint(0, len(self.scenario_paths) - 1)
    else:
        self.scenario_index += 1
        if self.scenario_index >= len(self.scenario_paths):
            self.scenario_index = 0
    self.current_scenario_path = self.scenario_paths[self.scenario_index]
    if self.remove_scenarios:
        self.scenario_paths.pop(self.scenario_index)

    target = next(p for p in self.scenario_paths if "ZAM_Tutorial-1_2_T-1" in p.name)
    self.current_scenario_path = target

    if not self.pickle_scenarios:
        self.simulation = Simulation(scenario_path=self.current_scenario_path, agent_config=self.agent_config,
                                     simulation_config=self.simulation_config,
                                     termination_config=self.termination_config)
    else:
        self.simulation = self._load_simulation_from_pickle(self.current_scenario_path)

    self.sim_agents = self.simulation.agents
    self.agent = self.sim_agents[0] if len(self.sim_agents) > 0 else None  # keep base reset from crashing

CommonRoadEnv.simulation_preprocessing = _patched_simulation_preprocessing

from safe_rl_envs.envs.commonroad.simulation.simulation import Simulation, Agent

def _patched_create_agents(self):
    agents = []
    for agent_id in self.agent_id_list:     # allow ANY number of agents
        agents.append(Agent(
            agend_id=agent_id,
            agent_config=self.agent_config,
            simulation_config=self.simulation_config,
            scenario=self.scenario,
            planning_problem=self.planning_problem_set.planning_problem_dict[agent_id]
        ))
    return agents

Simulation._create_agents = _patched_create_agents