from numba.cuda.tests.cudapy.test_intrinsics import intrinsic_forloop_step
from pettingzoo import ParallelEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers
import numpy as np

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from tianshou.env import PettingZooEnv
from multiagent_env import MultiAgentCommonRoadEnv
from gymnasium.envs.registration import register


class CRPettingZoo(ParallelEnv):

    """
    ParallelEnv wrapper for the underlying environment, later converted to an
    Agent-Environment-Cycle (AEC) environment for multi-agent scripts.
    """

    metadata = {
        "name": "CRPettingZoo",
        "render_mode": ["human"]
    }

    def __init__(self, raw_env):

        self._env = raw_env
        self.stepcounter = 0

        self.render_mode = getattr(self._env, "render_mode", None)

        self.agents = list(self._env.agent_ids)
        self.possible_agents = self.agents.copy()

        self.observation_spaces = {
            a: self._env.observation_space.spaces[a] for a in self.agents
        }

        self.action_spaces = {
            a: self._env.action_space.spaces[a] for a in self.agents
        }

        self._pending_dead = set()

    def reset(self, *, seed=None, options=None):

        res = self._env.reset(seed=seed, options=options)

        if isinstance(res, tuple) and len(res) == 2:
            full_obs, raw_info = res
        else:
            full_obs, raw_info = res, {}

        #agents alive at t=0
        self.agents = [a for a in self.possible_agents if a in full_obs]

        self._pending_dead.clear()

        obs = {a: full_obs[a] for a in self.agents}
        info = {
            a: (dict(raw_info[a]) if isinstance(raw_info, dict) and a in raw_info else {})
            for a in self.agents
        }

        #agent individual step counter
        self.agent_alive_steps = {a: 0 for a in self.agents}


        return obs, info

    def step(self, action_dict):

        self.stepcounter += 1

        try:
            full_obs, rd, dd, td, idict = self._env.step(action_dict)

        except Exception as e:
            import traceback
            traceback.print_exc()

        if not hasattr(self, "_last_obs"):
            self._last_obs = {}
        for a, o in full_obs.items():
            self._last_obs[a] = o

        #coerce to per-agent scalars/bools
        def _scal(x, a, default=0.0):
            v = x.get(a, default)
            return float(np.asarray(v).reshape(()))  # scalar float

        reward = {a: _scal(rd, a, 0.0) for a in self.possible_agents if a in rd}
        terminated = {a: bool(dd.get(a, False)) or a not in self.agents for a in self.possible_agents}
        truncated = {a: bool(td.get(a, False)) for a in self.possible_agents}
        info = {a: dict(idict.get(a, {})) for a in self.possible_agents}

        #compute from the current agent roster
        newly_dead = {a for a in self.agents if dd.get(a, False) or td.get(a, False)}
        alive_next = [a for a in self.agents if a not in newly_dead]

        self.agents = alive_next

        #on death step, include newly_dead in obs to avoid error
        obs_keys = set(alive_next) | newly_dead
        obs = {a: (full_obs.get(a, self._last_obs.get(a))) for a in obs_keys}

        for a in set(alive_next) | set(newly_dead):
            self.agent_alive_steps[a] = self.agent_alive_steps.get(a, 0) + 1

        self.last_reason = None
        for a in newly_dead:
            id_a = idict.get(a, {})
            tinfo = id_a.get("termination_info", {})
            reason = id_a.get("termination_reason", None)

            if reason is None:
                if tinfo.get("is_collision", 0):
                    reason = "collision"
                elif tinfo.get("is_off_road", 0):
                    reason = "offroad"
                elif tinfo.get("is_goal_reached_success", 0):
                    reason = "goal"
                elif tinfo.get("is_time_out", 0):
                    reason = "timeout"
                else:
                    reason = "unknown"

            self.last_reason = reason

        return obs, reward, terminated, truncated, info


    def action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def close(self):
        return self._env.close()

