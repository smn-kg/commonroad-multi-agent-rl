import os
import datetime
import traceback
from typing import Any
import numpy as np
import sys

# 1) Import the real CommonRoadEnv class from its original module
from safe_rl_envs.envs.commonroad.commonroad_env import CommonRoadEnv
from safe_rl_envs.envs.commonroad.simulation.simulation import Simulation
import random

#CC override of step function from CommonRoad Env
def _patched_step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:

    self.profiler.start("step_env_total")

    try:
        self.agent.planner_interface.set_action(action)

        self.profiler.start("step_simulation")
        self.status = self.simulation.step_sequential_simulation()
        self.profiler.stop("step_simulation")

        self.profiler.start("observation_collection")
        observation = self.oc.observe(self.agent)
        self.profiler.stop("observation_collection")

        self.profiler.start("calculate_reward")
        reward, reward_dict = self.reward_function.calc_reward(agent=self.agent,
                                                               observation_collector=self.oc)
        self.profiler.stop("calculate_reward")

        done, reason, termination_info = self.termination.is_terminated(agent=self.agent,
                                                                        observation_collector=self.oc)

        if reason is not None:
            self.termination_reason = reason

        if done:
            terminated = True
        else:
            terminated = False

        # in case termination is false, but agent returns termination, set termination to true
        if not self.status and not done:
            terminated = True
            self.termination_reason = ["simulation terminated"]

    except Exception:
        exc_info = traceback.format_exc()
        print(exc_info, file=sys.stderr)

        if self.logging_path is not None:
            exception_filename = "exception_log.txt"
            os.makedirs(self.logging_path, exist_ok=True)
            with open(os.path.join(self.logging_path, exception_filename), "a") as file:
                file.write(f"Exception occurred on {datetime.datetime.now()}\n")
                file.write(f"Scenario executed was: {str(self.agent.scenario.scenario_id)}\n")
                file.write(exc_info)
                file.write("\n\n")
                file.write(f"Action = {action}, timestep = {self.simulation.global_timestep}\n")
                file.write("\n\n")
                file.write(f"Agent state at the time of the exception:\n{self.agent.current_state}\n")
                file.write("\n\n")
                file.write(f"Observation dict at the time of the exception:\n{self.oc.observation_dict}\n")
                file.write("-" * 80 + "\n\n")

        observation = self.oc.observation_dict
        terminated = True
        reward = self.sparse_reward_config.reward_exception

        termination_info = {}
        self.termination_reason = ["exception occurred"]
        reward_dict = self.reward_function.empty_reward_dict()

    info = {
        # CC: scenario_name removed here
        "chosen_action": action,
        "current_ego_state": self.agent.current_state,
        "current_episode_time_step": self.simulation.global_timestep,
        "max_episode_time_steps": self.agent.agent_state.goal_checker.last_goal_timestep,
        "termination_info": termination_info,
        "termination_reason": ", ".join(self.termination_reason)
        if self.termination_reason is not None else None,
        "observation_dict": self.oc.observation_dict,
        "reward_dict": reward_dict,
    }

    if self.evaluation_mode:
        self.env_logger.log_step(self.current_uuid,
                                 self.simulation.global_timestep,
                                 action,
                                 reward_dict)

    self.current_timestep += 1
    truncated = False
    self.profiler.stop("step_env_total")

    return observation, reward, terminated, truncated, info


#CC override of simulation_preprocessing function
def _patched_simulation_preprocessing(self) -> None:
    #CC simulation override - removed single agent restriction
    if self.pick_random_scenario:
        self.scenario_index = random.randint(0, len(self.scenario_paths) - 1)
    else:
        self.scenario_index += 1
        if self.scenario_index >= len(self.scenario_paths):
            self.scenario_index = 0

    self.current_scenario_path = self.scenario_paths[self.scenario_index]
    if self.remove_scenarios:
        self.scenario_paths.pop(self.scenario_index)

    # force specific tutorial scenario
    target = next(p for p in self.scenario_paths
                  if "ZAM_Tutorial-1_2_T-1" in p.name)
    self.current_scenario_path = target

    if not self.pickle_scenarios:
        self.simulation = Simulation(
            scenario_path=self.current_scenario_path,
            agent_config=self.agent_config,
            simulation_config=self.simulation_config,
            termination_config=self.termination_config,
        )
    else:
        self.simulation = self._load_simulation_from_pickle(self.current_scenario_path)

    print("PATCHED SIM PREPROCESS USED")

    self.sim_agents = self.simulation.agents
    self.agent = self.sim_agents[0] if len(self.sim_agents) > 0 else None  # keep base reset from crashing