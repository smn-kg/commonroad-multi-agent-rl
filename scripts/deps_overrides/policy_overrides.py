# ppo_override.py
from __future__ import annotations

from typing import Any
import torch

from tianshou.policy.modelfree.ppo import PPOTrainingStats
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol

import numpy as np
from typing import cast

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy.multiagent.mapolicy import (
    MAPRolloutBatchProtocol,
)
from tianshou.data import Batch as BatchProtocol  # for isinstance check

from typing import cast

from tianshou.data import ReplayBuffer#,# RolloutBatchProtocol
from tianshou.data.types import BatchWithAdvantagesProtocol
from tianshou.data.utils.converter import to_torch_as


def _patched_ppo_learn(
    self,
    batch: RolloutBatchProtocol,
    batch_size: int | None,
    repeat: int,
    *args: Any,
    **kwargs: Any,
) -> PPOTrainingStats:
    losses, clip_losses, vf_losses, ent_losses = [], [], [], []
    gradient_steps = 0
    split_batch_size = batch_size or -1

    for step in range(repeat):
        if self.recompute_adv and step > 0:
            batch = self._compute_returns(batch, self._buffer, self._indices)

        for minibatch in batch.split(split_batch_size, merge_last=True):
            gradient_steps += 1
            advantages = minibatch.adv

            dist = self(minibatch).dist

            self.optim.zero_grad()

            if self.norm_adv:
                mean, std = advantages.mean(), advantages.std()
                advantages = (advantages - mean) / (std + self._eps)

            ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
            ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)

            surr1 = ratios * advantages
            surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

            if self.dual_clip:
                clip1 = torch.min(surr1, surr2)
                clip2 = torch.max(clip1, self.dual_clip * advantages)
                clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
            else:
                clip_loss = -torch.min(surr1, surr2).mean()

            # ---- GLOBAL CRITIC CHANGE ----
            value = self.critic(minibatch.obs_global).flatten()
            # ------------------------------

            if self.value_clip:
                v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(-self.eps_clip, self.eps_clip)
                vf1 = (minibatch.returns - value).pow(2)
                vf2 = (minibatch.returns - v_clip).pow(2)
                vf_loss = torch.max(vf1, vf2).mean()
            else:
                vf_loss = (minibatch.returns - value).pow(2).mean()

            ent_loss = dist.entropy().mean()
            loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

            loss.backward()
            self.optim.step()

            clip_losses.append(clip_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(ent_loss.item())
            losses.append(loss.item())

    return PPOTrainingStats.from_sequences(
        losses=losses,
        clip_losses=clip_losses,
        vf_losses=vf_losses,
        ent_losses=ent_losses,
        gradient_steps=gradient_steps,
    )

# a2c_override.py


def _patched_a2c_compute_returns(
    self,
    batch: RolloutBatchProtocol,
    buffer: ReplayBuffer,
    indices: np.ndarray,
) -> BatchWithAdvantagesProtocol:
    v_s, v_s_ = [], []
    with torch.no_grad():
        for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
            # ---- GLOBAL CRITIC CHANGE ----
            v_s.append(self.critic(minibatch.obs_global))
            v_s_.append(self.critic(minibatch.obs_global_next))
            # ------------------------------

    batch.v_s = torch.cat(v_s, dim=0).flatten()
    v_s_np = batch.v_s.cpu().numpy()
    v_s__np = torch.cat(v_s_, dim=0).flatten().cpu().numpy()

    if self.rew_norm:
        v_s_np = v_s_np * np.sqrt(self.ret_rms.var + self._eps)
        v_s__np = v_s__np * np.sqrt(self.ret_rms.var + self._eps)

    unnormalized_returns, advantages = self.compute_episodic_return(
        batch,
        buffer,
        indices,
        v_s__np,
        v_s_np,
        gamma=self.gamma,
        gae_lambda=self.gae_lambda,
    )

    if self.rew_norm:
        batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
        self.ret_rms.update(unnormalized_returns)
    else:
        batch.returns = unnormalized_returns

    batch.returns = to_torch_as(batch.returns, batch.v_s)
    batch.adv = to_torch_as(advantages, batch.v_s)
    return cast(BatchWithAdvantagesProtocol, batch)


def _patched_mapolicy_process_fn(  # type: ignore
        self,
        batch: MAPRolloutBatchProtocol,
        buffer: ReplayBuffer,
        indice: np.ndarray,
) -> MAPRolloutBatchProtocol:
    """Dispatch batch data from `obs.agent_id` to every policy's process_fn.

    Save original multi-dimensional rew in "save_rew", set rew to the
    reward of each agent during their "process_fn", and restore the
    original reward afterwards.
    """
    import numpy as np
    # TODO: maybe only str is actually allowed as agent_id? See MAPRolloutBatchProtocol
    results: dict[str | int, RolloutBatchProtocol] = {}
    assert isinstance(
        batch.obs,
        BatchProtocol,
    ), f"here only observations of type Batch are permitted, but got {type(batch.obs)}"
    # reward can be empty Batch (after initial reset) or nparray.
    has_rew = isinstance(buffer.rew, np.ndarray)
    if has_rew:  # save the original reward in save_rew
        # Since we do not override buffer.__setattr__, here we use _meta to
        # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
        save_rew, buffer._meta.rew = buffer.rew, Batch()  # type: ignore
    for agent, policy in self.policies.items():
        agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
        # agent_index = np.nonzero(batch.obs_next.agent_id == agent)[0]           ###
        if len(agent_index) == 0:
            results[agent] = cast(RolloutBatchProtocol, Batch())
            continue
        tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]

        # --- CC: carry global obs through MAPolicy slicing ---
        if hasattr(batch, "obs_global"):
            tmp_batch.obs_global = batch.obs_global[agent_index]
        if hasattr(batch, "obs_global_next"):
            tmp_batch.obs_global_next = batch.obs_global_next[agent_index]

        if has_rew:
            tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
            buffer._meta.rew = save_rew[:, self.agent_idx[agent]]

        if not hasattr(tmp_batch.obs, "mask"):
            if hasattr(tmp_batch.obs, "obs"):
                tmp_batch.obs = tmp_batch.obs.obs
            if hasattr(tmp_batch.obs_next, "obs"):
                tmp_batch.obs_next = tmp_batch.obs_next.obs

        results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)

    if has_rew:  # restore from save_rew
        buffer._meta.rew = save_rew
    return cast(MAPRolloutBatchProtocol, Batch(results))
