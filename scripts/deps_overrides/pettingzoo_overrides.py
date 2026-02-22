import copy
from collections import defaultdict
from typing import Callable, Dict, Optional

from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType, ParallelEnv
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

import warnings
from abc import ABC
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pettingzoo
from gymnasium import spaces
from packaging import version
from pettingzoo.utils.wrappers import BaseWrapper



#PETTINGZOO PARALLEL TO AEC ENV WRAPPER
class patched_parallel_to_aec_wrapper(AECEnv[AgentID, ObsType, Optional[ActionType]]):
    """Converts a Parallel environment into an AEC environment."""

    def __init__(
        self, parallel_env: ParallelEnv[AgentID, ObsType, Optional[ActionType]]
    ):
        self.env = parallel_env

        self.metadata = {**parallel_env.metadata}
        self.metadata["is_parallelizable"] = True

        try:
            self.render_mode = (
                self.env.render_mode  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            warnings.warn(
                f"The base environment `{parallel_env}` does not have a `render_mode` defined."
            )

        try:
            self.possible_agents = parallel_env.possible_agents
        except AttributeError:
            pass

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = (
                self.env.state_space  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            pass

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def observation_spaces(self):
        warnings.warn(
            "The `observation_spaces` dictionary is deprecated. Use the `observation_space` function instead."
        )
        try:
            return {
                agent: self.observation_space(agent) for agent in self.possible_agents
            }
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an `observation_spaces` dict attribute. Use the environments `observation_space` method instead"
            ) from e

    @property
    def action_spaces(self):
        warnings.warn(
            "The `action_spaces` dictionary is deprecated. Use the `action_space` function instead."
        )
        try:
            return {agent: self.action_space(agent) for agent in self.possible_agents}
        except AttributeError as e:
            raise AttributeError(
                "The base environment does not have an action_spaces dict attribute. Use the environments `action_space` method instead"
            ) from e

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        self._observations, self.infos = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents[:]
        self._live_agents = self.agents[:]
        self._actions: Dict[AgentID, Optional[ActionType]] = {
            agent: None for agent in self.agents
        }
        self._agent_selector = agent_selector(self._live_agents)
        self.agent_selection = self._agent_selector.reset()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        # Every environment needs to return infos that contain self.agents as their keys
        if not self.infos:
            warnings.warn(
                "The `infos` dictionary returned by `env.reset` was empty. OverwritingAgent IDs will be used as keys"
            )
            self.infos = {agent: {} for agent in self.agents}
        elif set(self.infos.keys()) != set(self.agents):
            self.infos = {agent: {self.infos.copy()} for agent in self.agents}
            warnings.warn(
                f"The `infos` dictionary returned by `env.reset()` is not valid: must contain keys for each agent defined in self.agents: {self.agents}. Overwriting with current info duplicated for each agent: {self.infos}"
            )

        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.new_agents = []
        self.new_values = {}

    def observe(self, agent):
        return self._observations[agent]

    def state(self):
        return self.env.state()

    def add_new_agent(self, new_agent):
        self._agent_selector._current_agent = len(self._agent_selector.agent_order)
        self._agent_selector.agent_order.append(new_agent)
        self.agent_selection = self._agent_selector.next()
        self.agents.append(new_agent)
        self.terminations[new_agent] = False
        self.truncations[new_agent] = False
        self.infos[new_agent] = {}
        self.rewards[new_agent] = 0
        self._cumulative_rewards[new_agent] = 0

    def step(self, action: Optional[ActionType]):
        #CC termination logic
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            del self._actions[self.agent_selection]
            assert action is None
            self._was_dead_step(action)
            return

        self._actions[self.agent_selection] = action
        #CC if last agent, step underlying env. Otherwise advance selector to next agent
        if self._agent_selector.is_last():
            obss, rews, terminations, truncations, infos = self.env.step(self._actions)

            self._observations = copy.copy(obss)
            self.terminations = copy.copy(terminations)
            self.truncations = copy.copy(truncations)
            self.infos = copy.copy(infos)
            self.rewards = copy.copy(rews)
            self._cumulative_rewards = copy.copy(rews)

            env_agent_set = set(self.env.agents)

            self.agents = self.env.agents + [
                agent
                for agent in sorted(self._observations.keys(), key=lambda x: str(x))
                if agent not in env_agent_set
            ]

            if len(self.env.agents):
                self._agent_selector = agent_selector(self.env.agents)
                self.agent_selection = self._agent_selector.reset()

            self._deads_step_first()
        else:
            if self._agent_selector.is_first():
                self._clear_rewards()

            self.agent_selection = self._agent_selector.next()

    def last(self, observe=True):
        agent = self.agent_selection

        observation = self.observe(agent) if observe else None

        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def __str__(self):
        return str(self.env)


#PETTINGZOO OVERRIDE
def _patched_was_dead_step(self, action: ActionType) -> None:
    """Helper function that performs step() for dead agents.

    Does the following:

    1. Removes dead agent from .agents, .terminations, .truncations, .rewards, ._cumulative_rewards, and .infos
    2. Loads next agent into .agent_selection: if another agent is dead, loads that one, otherwise load next live agent
    3. Clear the rewards dict

    Examples:
        Highly recommended to use at the beginning of step as follows:

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step()
            return
        # main contents of step
    """
    if action is not None:
        raise ValueError("when an agent is dead, the only valid action is None")

    # removes dead agent
    agent = self.agent_selection
    assert (
        self.terminations[agent] or self.truncations[agent]
    ), "an agent that was not dead as attempted to be removed"
    del self.terminations[agent]
    del self.truncations[agent]
    del self.rewards[agent]
    del self._cumulative_rewards[agent]
    del self.infos[agent]
    self.agents.remove(agent)

    #CC pick a valid next selection from current roster
    still_dead = [a for a in self.agents
                  if self.terminations.get(a, False) or self.truncations.get(a, False)]

    if still_dead:
        next_sel = still_dead[0]  # chain remaining deads
    elif self.agents:
        next_sel = self.agents[0]  # first alive
    else:
        next_sel = None  # episode over

    self._skip_agent_selection = None
    self.agent_selection = next_sel

    #CC resync selector
    if self.agent_selection is not None:
        from pettingzoo.utils import agent_selector
        order = list(self.agents)
        self._agent_selector = agent_selector(order)
        self._agent_selector._current_agent = order.index(self.agent_selection)

    if hasattr(self, "_live_agents"):
        self._live_agents = list(self.agents)


    self._clear_rewards()


#PETTINGZOO OVERRIDE
def _patched_deads_step_first(self) -> AgentID:
    """Makes .agent_selection point to first terminated agent.

    Stores old value of agent_selection so that _was_dead_step can restore the variable after the dead agent steps.
    """

    # CC get terminated agents
    _deads_order = [
        agent
        for agent in self.agents
        if (self.terminations[agent] or self.truncations[agent])
    ]
    if _deads_order:
        self._skip_agent_selection = self.agent_selection
        self.agent_selection = _deads_order[0]

        # CC aligning the selector's cursor to the forced selection
        order = getattr(self._agent_selector, "agent_order", list(self.agents))
        if self.agent_selection in order:
            self._agent_selector._current_agent = order.index(self.agent_selection)
        else:
            # CC fallback: rebuild selector from current roster
            from pettingzoo.utils import agent_selector
            order = list(self.agents)
            self._agent_selector = agent_selector(order)
            self._agent_selector._current_agent = order.index(self.agent_selection)

    return self.agent_selection

#TIANSHOU PETTINGZOO OVERRIDE

if version.parse(pettingzoo.__version__) < version.parse("1.21.0"):
    warnings.warn(
        f"You are using PettingZoo {pettingzoo.__version__}. "
        f"Future tianshou versions may not support PettingZoo<1.21.0. "
        f"Consider upgrading your PettingZoo version.",
        DeprecationWarning,
    )

BASE_DIR = Path(__file__).resolve().parents[1]
#MODULE FROM FILE LOADER
def load_module_from_file(file_path: Path) -> ModuleType:
    # Create a fresh module
    module = ModuleType(file_path.stem)
    # Read and execute its source into the module’s namespace
    code = file_path.read_text()
    exec(compile(code, str(file_path), "exec"), module.__dict__)
    return module

#CC Load multiagent_env.py as a module
ep_met = load_module_from_file(BASE_DIR / "episode_metrics.py")

class PettingZooEnv(AECEnv, ABC):
    """The interface for petting zoo environments.

    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.PettingZooEnv`. Here is the usage:
    ::

        env = PettingZooEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, trunc, term, info = env.step(action)
        env.close()

    The available action's mask is set to True, otherwise it is set to False.
    Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self, env: BaseWrapper):
        super().__init__()
        self.env = env
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.rewards = [0] * len(self.agents)

        #CC Get first observation space
        self.observation_space: Any = self.env.observation_space(self.agents[0])

        #CC Get first action space
        self.action_space: Any = self.env.action_space(self.agents[0])

        assert all(
            self.env.observation_space(agent) == self.observation_space for agent in self.agents
        ), (
            "Observation spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_observations wrapper can help (usage: "
            "`supersuit.pad_observations_v0(env)`"
        )

        assert all(self.env.action_space(agent) == self.action_space for agent in self.agents), (
            "Action spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_action_space wrapper can help (usage: "
            "`supersuit.pad_action_space_v0(env)`"
        )

        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        self.env.reset(*args, **kwargs)

        observation, reward, terminated, truncated, info = self.env.last()#self

        observation_dict = {
            "agent_id": self.env.agent_selection,
            "obs": observation,
        }

        return observation_dict, info

    def _s(self, o):
        import numpy as np
        a = np.asarray(o, np.float32)
        return f"shape={a.shape} μ={a.mean():+.2f} σ={a.std():.2f}"

    def step(self, action: Any) -> tuple[dict, list[int], bool, bool, dict]:

        #CC current agent in AEC
        cur = self.env.agent_selection
        is_dead = self.env.terminations.get(cur, False) or self.env.truncations.get(cur, False)

        #CC action for AEC dead step after an agent terminates
        aec_action = None if is_dead else action

        #CC step underlying AEC wrapper/env
        self.env.step(aec_action)

        #CC retrieve data tuple from AEC last
        observation, rew, term, trunc, info = self.env.last()

        obs = {"agent_id": self.env.agent_selection, "obs": observation}

        idx = self.agent_idx[self.env.agent_selection]
        rew_vec = np.zeros(len(self.agent_idx), dtype=np.float32)
        rew_vec[idx] = float(rew)

        inner_env = getattr(self.env, "env", None)  #OrderEnforcingWrapper
        inner_inner_env = getattr(inner_env, "env", None)  #parallel_to_aec_wrapper
        base_env = getattr(inner_inner_env, "env", None)  #ParallelEnv

        #CC termintation reason
        reason = getattr(base_env, "last_reason", None) if base_env else None

        #CC log data to episode metrics
        ep_met.on_env_step(rew_vec, term, trunc, reason=reason, agent_id= self.env.agent_selection)

        return obs, rew_vec, term, trunc, info

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except (NotImplementedError, AttributeError):
            self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()

    def get_agent_idx(self):
        """Expose agent index mapping to outer wrappers or collectors."""                           ###
        return self.agent_idx
