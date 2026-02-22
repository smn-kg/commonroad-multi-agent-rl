from dataclasses import dataclass
from types import ModuleType
from pathlib import Path
from typing import Callable, Any

from functools import wraps

import tianshou as ts
import torch
from safe_rl_envs.configs.env_configs import BaseEnvConfig
# from tianshou.env import PettingZooEnv
from deps_overrides.pettingzoo_overrides import PettingZooEnv
from torch import nn

from pettingzoo.utils import parallel_to_aec

from safe_rl_lib.data_models.model_configs import BaseTorchModelConfig
from safe_rl_lib.data_models.policy_configs import PolicyConfig
from safe_rl_lib.data_models.trainer_configs import (
    BaseTrainerConfig,
    OnPolicyTrainerConfig,
    OffPolicyTrainerConfig,
)

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from pettingzoo.utils.wrappers import BaseWrapper

import numpy as np, gymnasium as gym
import time, types


import warnings
warnings.simplefilter("default")

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="ep_return should be a scalar but is a numpy array: self._ep_return.shape=(2,).*"
)

#import modules form other python files
def load_module_from_file(file_path: Path) -> ModuleType:
    module = ModuleType(file_path.stem)
    code = file_path.read_text()
    exec(compile(code, str(file_path), "exec"), module.__dict__)
    return module


BASE_DIR = Path(__file__).resolve().parents[1]

ma_mod = load_module_from_file(BASE_DIR / "multiagent_env.py")
MultiAgentCommonRoadEnv = ma_mod.MultiAgentCommonRoadEnv

pz_mod = load_module_from_file(BASE_DIR / "parallel_env.py")
CRPettingZoo = pz_mod.CRPettingZoo

temp_norm = load_module_from_file(BASE_DIR / "obs_stats.py")
GLOBAL_OBS_STAT = temp_norm.GLOBAL_OBS_STAT

ep_met = load_module_from_file(BASE_DIR / "episode_metrics.py")


# WRAPPERS

# intercepts and scales rewards
class RewardScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale
    def reward(self, rew):
        import numpy as np
        return np.asarray(rew, dtype=np.float32) * self.scale

# normalizes obs using Z-Score normalization
class AECObsNormalize(BaseWrapper):
    """Normalize observations for PettingZoo AEC envs via .observe() / .last()."""
    def __init__(self, env, stat, update: bool = True, clip: float | None = 5.0):
        super().__init__(env)
        self.stat = stat
        self.update = update
        self.clip = clip

        self._upd_hits = 0

    # PettingZoo AEC API
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def _s(self, a):
        import numpy as np
        a = np.asarray(a, np.float32)
        return f"shape={a.shape} μ={a.mean():+.2f} σ={a.std():.2f}"

    # Normalize per-agent observe()
    def observe(self, agent):
        x = self.env.observe(agent)
        if x is None:
            return None
        x = np.asarray(x, dtype=np.float32)
        if self.update:
            self.stat.update(x)

        z = self.stat.normalize(x, clip=self.clip)

        print(f"[AECNorm→PZ.observe] {self._s(z)}")
        return z

    # Normalize obs returned by last()
    def last(self, observe=True):
        obs, rew, term, trunc, info = self.env.last(observe=observe)
        if obs is None:
            ag = getattr(self.env, "agent_selection", None)
            if ag is not None:
                obs = self.env.observe(ag)


        if obs is not None:
            obs = np.asarray(obs, dtype=np.float32)
            if self.update:
                self.stat.update(obs)

            z = self.stat.normalize(obs, clip=self.clip)

            obs = z
        return obs, rew, term, trunc, info


@dataclass
class SetupPipeline:
    policy_config: PolicyConfig
    model_config: BaseTorchModelConfig
    env_config: BaseEnvConfig
    n_agents: int
    train_config: BaseTrainerConfig | None = None
    logger: ts.utils.BaseLogger | None = None


    env_wrapper_function: Callable[[gym.Env], gym.Env] | None = None
    test_env_wrapper_function: Callable[[gym.Env], gym.Env] | None = None

    def _create_models(self, env: gym.Env) -> nn.Module:
        raise NotImplementedError

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.model_config.lr)


    def _env_function(self):

        raw_env = gym.make(
            self.env_config.env_name,
            render_mode="human",
            **self.env_config.model_dump(exclude={"env_name", "render_mode"})
        )
        pz_aec = parallel_to_aec(CRPettingZoo(raw_env))
        pz_aec = AECObsNormalize(pz_aec, stat=GLOBAL_OBS_STAT, update=True, clip=5.0)
        ts_env = PettingZooEnv(pz_aec)
        ts_env = RewardScale(ts_env, scale=0.1)
        return ts_env

    #CC testing functionality, not guaranteed to be working
    def _env_function_test(self):
        raw_env = gym.make(
            self.env_config.env_name,
            render_mode="human",
            **self.env_config.model_dump(exclude={"env_name", "render_mode"})
        )
        pz_aec = parallel_to_aec(CRPettingZoo(raw_env))
        pz_aec = AECObsNormalize(pz_aec, stat=GLOBAL_OBS_STAT, update=False, clip=5.0)
        ts_env = PettingZooEnv(pz_aec)
        ts_env = RewardScale(ts_env, scale=0.1)
        return ts_env

    def _create_environments(self) -> tuple[ts.env.DummyVectorEnv, ts.env.DummyVectorEnv]:
        train_envs = ts.env.DummyVectorEnv(
            [self._env_function for _ in range(self.train_config.n_train_envs)]
        )
        test_envs = ts.env.DummyVectorEnv(
            [self._env_function for _ in range(self.train_config.n_test_envs)]  #test
        )
        return train_envs, test_envs

    def _create_policy(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, env: gym.Env
    ) -> ts.policy.BasePolicy:
        raise NotImplementedError


    def create_policy_for_eval(self, env: gym.Env) -> ts.policy.BasePolicy:
        model = self._create_models(env)
        optimizer = self._create_optimizer(model)
        policy = self._create_policy(model, optimizer, env)
        return policy

    def create_single_env(self) -> PettingZooEnv:
        return self._env_function()

    def _create_collector(
        self,
        policy: ts.policy.BasePolicy,
        train_envs: ts.env.DummyVectorEnv,
        test_envs: ts.env.DummyVectorEnv,
    ) -> [ts.data.Collector, ts.data.Collector]:
        train_collector = ts.data.Collector(
            policy,
            train_envs,
            ts.data.VectorReplayBuffer(
                self.train_config.buffer_size, self.train_config.n_train_envs
            ),
        )
        #CC CENTRAL CRITIC FIX
        buf = train_collector.buffer
        extra = {"obs_global", "obs_global_next"}

        # allow manager to keep these keys
        buf._reserved_keys = tuple(set(buf._reserved_keys) | extra)

        # allow each child ReplayBuffer to accept them in set_batch
        for b in buf.buffers:
            b._reserved_keys = tuple(set(b._reserved_keys) | extra)


        test_collector = ts.data.Collector(policy, test_envs,)

        return train_collector, test_collector


    def create_trainer(self, logdir: str, purge_step: int = 0,
                           writer: SummaryWriter | None = None,
                           logger: TensorboardLogger | None = None,
                           step_offset=0):
        assert (
            self.train_config is not None
        ), 'Train config must be provided when creating a trainer.'

        train_envs, test_envs = self._create_environments()

        env0 = train_envs.workers[0].env

        n_agents = len(env0.agents)
        print("Detected agents:", env0.agents)

        policies = []
        for _ in range(n_agents):
            model = self._create_models(env0)
            optim = self._create_optimizer(model)
            pol = self._create_policy(model, optim, env0)

            policies.append(pol)

        policy = ts.policy.MultiAgentPolicyManager(policies=policies, env=env0)

        policy.action_scaling = True
        policy.action_bound_method = "tanh"


        #CC log loss stats
        orig_update = policy.update

        def _tapped_update(*a, **k):
            out = orig_update(*a, **k) or {}
            policy._last_raw = (out.get_loss_stats_dict() if hasattr(out, "get_loss_stats_dict") else out)
            return out

        policy.update = _tapped_update

        self._last_update_info = None

        def _wrap_ret_capture(orig):
            @wraps(orig)
            def wrapped(*a, **kw):
                out = orig(*a, **kw)  # call the real method
                info = out if isinstance(out, dict) else getattr(out, "__dict__", None)
                if info:
                    self._last_update_info = info
                return out

            return wrapped

        for name in ("learn", "update"):
            if hasattr(policy, name):
                orig = getattr(policy, name)
                setattr(policy, name, _wrap_ret_capture(orig))



        #CC monitor policy forward and inject obs_next if missing to prevent error
        '''truncates displayed obs Batches'''
        def _head(x, n=8):
            try:
                a = np.asarray(x)
                flat = a.reshape(-1)
                return f"shape={tuple(a.shape)} head={flat[:n].tolist()}"
            except Exception:
                return f"type={type(x)}"

        _orig_map_forward = policy.forward

        PRINT_EVERY = 10000

        _calls = {"n": 0}

        def _map_forward_with_obsnext_and_dbg(self, batch, state=None, **kw):
            _calls["n"] += 1

            if isinstance(batch, dict):
                for b in batch.values():
                    if not hasattr(b, "obs_next") or getattr(b, "obs_next", None) is None:
                        b.obs_next = b.obs
            else:
                if not hasattr(batch, "obs_next") or getattr(batch, "obs_next", None) is None:
                    batch.obs_next = batch.obs

            t0 = time.time()
            out = _orig_map_forward(batch, state=state, **kw)
            dt_ms = (time.time() - t0) * 1e3

            def _get_act(o):
                if hasattr(o, "act"):
                    return o.act
                if isinstance(o, dict) and "act" in o:
                    return o["act"]
                return o

            act = _get_act(out)

            if _calls["n"] % PRINT_EVERY == 0:
                if isinstance(batch, dict):
                    print(f"[MAPM] AEC turn #{_calls['n']} dt={dt_ms:.1f} ms agents={list(batch.keys())}")
                    for k, b in batch.items():
                        a = act[k] if isinstance(act, dict) else act
                        print(f"  └─ agent={k} obs({_head(b.obs)}) -> act={np.asarray(a)}")
                else:
                    #CC single agent turn (typical AEC step)
                    agent_id = None
                    if hasattr(batch, "agent_id"):
                        agent_id = getattr(batch, "agent_id")
                    elif hasattr(batch, "info") and isinstance(batch.info, dict):
                        agent_id = batch.info.get("agent_id")
                    print(f"[MAPM] AEC turn #{_calls['n']} dt={dt_ms:.1f} ms "
                          f"agent={agent_id} obs({_head(batch.obs)}) -> act={np.asarray(act)}")

            return out

        policy.forward = types.MethodType(_map_forward_with_obsnext_and_dbg, policy)

        train_collector, test_collector = self._create_collector(policy, train_envs, test_envs)
        train_collector.reset()


        #CC log metrics like goal-reach or collision rate
        _calls = {"n": 0}

        def train_fn(epoch, env_step):
            print("env_step:", env_step)
            writer.add_scalar("debug/heartbeat", env_step, env_step)

            d = getattr(policy, "_last_raw", None)
            print("[DBG] _last_raw type:", type(d))

            if isinstance(d, dict) and d:
                for k, v in d.items():
                    try:
                        if hasattr(v, "item") and getattr(v, "numel", lambda: 1)() == 1:
                            val = float(v.item())
                        else:
                            val = float(v)
                        writer.add_scalar(f"ppo/{k}", val, env_step)
                    except Exception:
                        print(f"[SKIP] non-scalar at key {k}: {type(v)}")
            else:
                print("[DBG] no stats yet (update hasn’t run or tap not applied)")


            buf = getattr(ep_met, "EP_BUF", None)
            if buf:
                print("[EP_BUF preview]", [buf[i] for i in range(min(len(buf), 5))])

            if buf and len(buf):
                data = []
                while buf:
                    data.append(buf.popleft())

                by_agent = {}
                for d in data:
                    aid = int(d["agent_id"])
                    by_agent.setdefault(aid, []).append(d)

                for aid, entries in by_agent.items():
                    E = np.array([e["return"] for e in entries])
                    steps = np.array([e["steps"] for e in entries])
                    reward_rate = E / np.maximum(steps, 1.0)

                    writer.add_scalar(f"metrics/return_mean_agent{aid}", float(E.mean()), env_step)
                    writer.add_scalar(f"metrics/reward_rate_mean_agent{aid}",
                                      float(reward_rate.mean()), env_step)

                    reasons = [str(e["reason"]).lower() for e in entries]
                    print("[reasons first20]", reasons[:20])
                    total = len(reasons) or 1

                    goal_rate = sum("goal" in r for r in reasons) / total
                    coll_rate = sum("collis" in r for r in reasons) / total
                    offroad_rate = sum("off" in r and "road" in r for r in reasons) / total

                    writer.add_scalar(f"metrics/goal_reached_rate_agent{aid}", goal_rate, env_step)
                    writer.add_scalar(f"metrics/is_collision_rate_agent{aid}", coll_rate, env_step)
                    writer.add_scalar(f"metrics/is_off_road_rate_agent{aid}", offroad_rate, env_step)

                writer.add_scalar("metrics/episode_len", float(steps.mean()), env_step)

                print("[TB] writer logdir =", writer.log_dir)

            writer.flush()

        def make_save_best_fn(
                get_norm_state_dict: Callable[[], dict[str, Any]] | None = None,
        ):
            save_dir = (Path(__file__).resolve().parent / "outputs" / "checkpoints")
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / "best_policy.pth"

            def save_best_fn(policy):
                payload = {
                    "model": policy.state_dict(),
                    "norm_obs": get_norm_state_dict(),  # e.g., obs_rms state dict or None
                }
                torch.save(payload, ckpt_path)
                print(f"[save_best_fn] wrote {ckpt_path}")

            return save_best_fn

        def get_norm_state_dict():
            return {
                "mean": GLOBAL_OBS_STAT.mean,
                "M2": GLOBAL_OBS_STAT.M2,
                "n": GLOBAL_OBS_STAT.n,
                "eps": GLOBAL_OBS_STAT.eps,
            }

        save_best_fn = make_save_best_fn(get_norm_state_dict=get_norm_state_dict)

        if isinstance(self.train_config, OnPolicyTrainerConfig):
            print("on policy branch entered")
            trainer = ts.trainer.OnpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                logger=self.logger,
                batch_size=self.model_config.batch_size,
                max_epoch=self.train_config.n_epochs,
                step_per_epoch=self.train_config.steps_per_epoch,
                repeat_per_collect=self.train_config.repeats_per_collect,
                episode_per_test=self.train_config.episodes_per_test,
                step_per_collect=self.train_config.steps_per_collect,
                train_fn=train_fn,
                test_in_train=False,
                save_best_fn=save_best_fn,
            )
        elif isinstance(self.train_config, OffPolicyTrainerConfig):
            print("off policy branch entered")
            trainer = ts.trainer.OffpolicyTrainer(
                policy=policy,
                train_collector=train_collector,
                test_collector=test_collector,
                logger=self.logger,
                batch_size=self.model_config.batch_size,
                max_epoch=self.train_config.n_epochs,
                step_per_epoch=self.train_config.steps_per_epoch,
                update_per_step=self.train_config.updates_per_step,
                episode_per_test=self.train_config.episodes_per_test,
                step_per_collect=self.train_config.steps_per_collect,
            )
        else:
            raise NotImplementedError

        return trainer

