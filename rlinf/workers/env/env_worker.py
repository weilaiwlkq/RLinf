# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RTCActionResponse,
    RTCRequest,
    RolloutResult,
    Trajectory,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import update_nested_cfg
from rlinf.utils.placement import HybridComponentPlacement


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self.rollout_epoch = self.cfg.algorithm.get("rollout_epoch", 1)
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        # Env configurations
        self.enable_offload = self.cfg.env.train.get("enable_offload", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.enable_eval = self.cfg.runner.val_check_interval > 0 or self.only_eval
        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.train_chunk_pause_seconds = float(
            self.cfg.env.train.get("chunk_pause_seconds", 0.0)
        )
        self.eval_chunk_pause_seconds = float(
            self.cfg.env.eval.get("chunk_pause_seconds", 0.0)
        )
        self.actor_split_num = self.get_actor_split_num()

    def init_worker(self):
        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }

        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
        self.log_info(f"Env worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Env worker initialized with src_ranks: {self.src_ranks}")
        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
        eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)

        if not self.only_eval:
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
        if self.enable_eval:
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )

        if not self.only_eval:
            self._init_env()

    def update_env_cfg(self):
        # train env
        train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
        if train_override_cfgs is not None:
            assert len(train_override_cfgs) > self._rank, (
                f"{len(train_override_cfgs)=} > {self._rank=}"
            )

            general_train_override_cfg = OmegaConf.to_container(
                self.cfg.env.train.get("override_cfg", {}), resolve=True
            )
            override_cfg = OmegaConf.to_container(
                train_override_cfgs[self._rank], resolve=True
            ).copy()

            base_cfg = {}
            base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
            base_cfg = update_nested_cfg(base_cfg, override_cfg)
            setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))

        eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
        if eval_override_cfgs is not None:
            assert len(eval_override_cfgs) > self._rank, (
                f"{len(eval_override_cfgs)=} > {self._rank=}"
            )

            general_eval_override_cfg = OmegaConf.to_container(
                self.cfg.env.eval.get("override_cfg", {}), resolve=True
            )
            eval_override_cfg = OmegaConf.to_container(
                eval_override_cfgs[self._rank], resolve=True
            ).copy()
            base_eval_cfg = {}
            base_eval_cfg = update_nested_cfg(base_eval_cfg, general_eval_override_cfg)
            base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
            setattr(self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    stats_sample_ratio=getattr(
                        env_cfg.data_collection, "stats_sample_ratio", 0.1
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                    save_debug_media=getattr(
                        env_cfg.data_collection, "save_debug_media", False
                    ),
                    debug_video_fps=getattr(
                        env_cfg.data_collection, "debug_video_fps", None
                    ),
                )
            env_list.append(env)
        return env_list

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout peer ranks for this env worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for both sending
        env outputs and receiving action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(rollout_rank, batch_size)`` tuples this env worker should send
            env outputs to.
        """
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute rollout source ranks and sizes for receiving action chunks."""
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            dst_rank=self._rank,
        )

    def _init_env(self):
        for i in range(self.stage_num):
            if self.cfg.env.train.auto_reset:
                extracted_obs, _ = self.env_list[i].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))
            if self.enable_offload and hasattr(self.env_list[i], "offload"):
                self.env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]
        if "gripper_debug" in infos:
            for key, value in infos["gripper_debug"].items():
                env_info[f"gripper_debug/{key}"] = value.cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        rewrite_chunk_gripper = bool(
            self.cfg.actor.model.get("rewrite_chunk_gripper", False)
        )
        if rewrite_chunk_gripper:
            # Chunk-level gripper rule:
            # if there are >=2 zeros in this chunk, set all gripper actions to 0;
            # otherwise set all gripper actions to 1.
            gripper = chunk_actions[..., -1]
            if isinstance(gripper, torch.Tensor):
                gripper_binary = (gripper > 0.5).to(dtype=gripper.dtype)
                zeros_count = (1.0 - gripper_binary).sum(dim=1, keepdim=True)
                final_gripper = (zeros_count < 2).to(dtype=gripper.dtype)
                chunk_actions[..., -1] = final_gripper
            else:
                gripper_np = np.asarray(gripper)
                gripper_binary = (gripper_np > 0.5).astype(gripper_np.dtype)
                zeros_count = (1.0 - gripper_binary).sum(axis=1, keepdims=True)
                final_gripper = (zeros_count < 2).astype(gripper_np.dtype)
                chunk_actions[..., -1] = final_gripper
        if isinstance(chunk_actions, torch.Tensor):
            action_wo_gripper = chunk_actions[..., :-1].detach().cpu().numpy()
        else:
            action_wo_gripper = np.asarray(chunk_actions)[..., :-1]
        print("[env_evaluate_step] action[..., :-1]:", action_wo_gripper)
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_env_list[stage_id].chunk_step(chunk_actions)
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        env_info["baseline_chunk_steps"] = torch.tensor(
            [chunk_rewards.shape[1]], dtype=torch.float32
        )

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()
        if "gripper_debug" in infos:
            for key, value in infos["gripper_debug"].items():
                env_info[f"gripper_debug/{key}"] = value.cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
        )
        return env_output, env_info

    def _assert_rtc_eval_supported(self):
        rtc_cfg = self.cfg.runner.get("rtc", {})
        if not rtc_cfg.get("enabled", False):
            return
        assert self.cfg.env.eval.env_type == "realworld", (
            "RTC evaluation is currently only integrated for real-world envs."
        )
        assert str(self.cfg.actor.model.model_type) == "openpi", (
            "RTC real-world evaluation is currently integrated for the OpenPI policy path."
        )
        assert self.stage_num == 1, (
            "RTC real-world evaluation currently supports a single pipeline stage."
        )
        assert self.eval_num_envs_per_stage == 1, (
            "RTC real-world evaluation currently supports a single env per worker."
        )

    def send_rtc_request(
        self, output_channel: Channel, rtc_request: RTCRequest, mode: str = "eval"
    ) -> None:
        assert mode == "eval", "RTC requests are only supported in eval mode."
        dst_ranks_and_sizes = self.dst_ranks[mode]
        assert len(dst_ranks_and_sizes) == 1, (
            "RTC real-world evaluation currently supports a single env->rollout route."
        )
        dst_rank, _ = dst_ranks_and_sizes[0]
        output_channel.put(
            item=rtc_request,
            key=CommMapper.build_channel_key(self._rank, dst_rank, extra=f"{mode}_rtc"),
        )

    def recv_rtc_response(
        self, input_channel: Channel, mode: str = "eval", async_op: bool = False
    ):
        assert mode == "eval", "RTC responses are only supported in eval mode."
        src_ranks_and_sizes = self.src_ranks[mode]
        assert len(src_ranks_and_sizes) == 1, (
            "RTC real-world evaluation currently supports a single rollout->env route."
        )
        src_rank, _ = src_ranks_and_sizes[0]
        return input_channel.get(
            key=CommMapper.build_channel_key(src_rank, self._rank, extra=f"{mode}_rtc"),
            async_op=async_op,
        )

    def _maybe_rewrite_eval_chunk_gripper(self, chunk_actions):
        rewrite_chunk_gripper = bool(
            self.cfg.actor.model.get("rewrite_chunk_gripper", False)
        )
        if not rewrite_chunk_gripper:
            return chunk_actions

        gripper = chunk_actions[..., -1]
        if isinstance(gripper, torch.Tensor):
            gripper_binary = (gripper > 0.5).to(dtype=gripper.dtype)
            zeros_count = (1.0 - gripper_binary).sum(dim=1, keepdim=True)
            final_gripper = (zeros_count < 2).to(dtype=gripper.dtype)
            chunk_actions[..., -1] = final_gripper
        else:
            gripper_np = np.asarray(gripper)
            gripper_binary = (gripper_np > 0.5).astype(gripper_np.dtype)
            zeros_count = (1.0 - gripper_binary).sum(axis=1, keepdims=True)
            final_gripper = (zeros_count < 2).astype(gripper_np.dtype)
            chunk_actions[..., -1] = final_gripper
        return chunk_actions

    def _copy_rtc_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.clone()
        if isinstance(obs, dict):
            return {key: self._copy_rtc_obs(value) for key, value in obs.items()}
        if isinstance(obs, list):
            return [self._copy_rtc_obs(value) for value in obs]
        if isinstance(obs, tuple):
            return tuple(self._copy_rtc_obs(value) for value in obs)
        return obs

    def env_evaluate_rtc_action(
        self, env_action: torch.Tensor | np.ndarray, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        if isinstance(env_action, torch.Tensor):
            action_wo_gripper = env_action[..., :-1].detach().cpu().numpy()
        else:
            action_wo_gripper = np.asarray(env_action)[..., :-1]
        print("[env_evaluate_rtc_action] action[..., :-1]:", action_wo_gripper)

        extracted_obs, step_reward, terminations, truncations, infos = self.eval_env_list[
            stage_id
        ].step(env_action, auto_reset=self.cfg.env.eval.auto_reset)

        env_info = {}
        dones = torch.logical_or(terminations, truncations)
        final_obs = (
            infos["final_observation"]
            if isinstance(infos, dict) and "final_observation" in infos
            else None
        )

        if isinstance(infos, dict):
            if dones.any():
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
                if "final_info" in infos:
                    final_info = infos["final_info"]
                    if "episode" in final_info:
                        for key in final_info["episode"]:
                            env_info[key] = final_info["episode"][key][dones].cpu()
            if "gripper_debug" in infos:
                for key, value in infos["gripper_debug"].items():
                    env_info[f"gripper_debug/{key}"] = value.cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=step_reward,
            dones=dones,
            terminations=terminations,
            truncations=truncations,
        )
        return env_output, env_info

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        """Receive and merge chunked actions for the current env worker.

        The method fetches one action shard from each mapped rollout source rank
        under a deterministic channel key pattern and concatenates them on the
        batch dimension.

        Args:
            input_channel: Channel carrying rollout->env action chunks.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            Concatenated action chunk array with shape ``[num_envs_per_stage, ...]``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )

            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video and isinstance(
                    self.env_list[i], RecordVideo
                ):
                    self.env_list[i].flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video and isinstance(
                    self.eval_env_list[i], RecordVideo
                ):
                    self.eval_env_list[i].flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    def split_env_batch(
        self,
        env_batch: dict[str, Any],
        sizes: list[int],
        mode: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        """Split one env batch dict into size-specified sub-batches along dim-0.

        Tensor values are chunked on dim-0; list values are sliced proportionally;
        nested dict values are split recursively.

        Args:
            env_batch: Env output dictionary produced by ``EnvOutput.to_dict``.
            sizes: Batch sizes for each destination rank.
            mode: Rollout mode used for list-length validation.

        Returns:
            A list of split env batches, one item per destination rank.
        """
        count = len(sizes)
        total_size = sum(sizes)
        splitted_env_batches = [{} for _ in range(count)]
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == total_size, (
                    f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
                )
                splitted_values = torch.split(value, sizes, dim=0)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_values[i].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage} "
                        f"(train_num_envs_per_stage), got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage} "
                        f"(eval_num_envs_per_stage), got {length}"
                    )
                assert length == total_size, (
                    f"List field '{key}' expected length {total_size}, got {length}."
                )
                begin = 0
                for i, size in enumerate(sizes):
                    splitted_env_batches[i][key] = value[begin : begin + size]
                    begin += size
            elif isinstance(value, dict):
                splitted_sub_batches = self.split_env_batch(value, sizes, mode)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_sub_batches[i]
            else:
                for i in range(count):
                    splitted_env_batches[i][key] = value

        return splitted_env_batches

    def send_env_batch(
        self,
        output_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Send split env batches to mapped rollout ranks.

        Each destination rank receives one split batch via a stable key built from
        ``src_rank``, ``dst_rank`` and ``mode``.

        Args:
            output_channel: Channel carrying env->rollout outputs.
            env_batch: Env output dictionary for one pipeline stage.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = self.split_env_batch(env_batch, split_sizes, mode)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            output_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    async def _run_interact_once(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                env_output: EnvOutput = env_outputs[stage_id]
                env_batch = env_output.to_dict()
                self.send_env_batch(
                    output_channel,
                    {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    },
                )

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None,
                        prev_values=rollout_result.prev_values
                        if self.collect_prev_infos
                        else None,
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    if self.train_chunk_pause_seconds > 0:
                        await asyncio.sleep(self.train_chunk_pause_seconds)
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values
                )
                chunk_step_result = ChunkStepResult(
                    prev_values=rollout_result.prev_values
                    if self.collect_prev_infos
                    else None,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            output_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.enable_offload and hasattr(env, "offload"):
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        eval_metrics = defaultdict(list)
        eval_metrics["baseline_episode_count"] = []
        total_action_chunks = 0
        total_env_steps = 0
        total_episode_time_s = 0.0
        stop_all_eval = False
        stop_eval_on_success = bool(
            self.cfg.runner.get("stop_eval_on_keyboard_success", False)
        )

        def _success_from_info(env_info: dict[str, Any]) -> bool:
            for key in ("success_once", "success_at_end", "success"):
                value = env_info.get(key)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    return bool(value.any().item())
                return bool(np.asarray(value).any())
            return False

        def _send_eval_obs(env_output: EnvOutput, eval_stop: bool = False):
            env_batch = env_output.to_dict()
            env_batch["eval_stop"] = eval_stop
            self.send_env_batch(
                output_channel,
                {
                    "obs": env_batch["obs"],
                    "final_obs": env_batch["final_obs"],
                    "eval_stop": env_batch["eval_stop"],
                },
                mode="eval",
            )

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    _send_eval_obs(env_output)

            episode_t0 = time.perf_counter()
            episode_action_chunks = 0
            episode_steps = 0
            episode_done = False
            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    chunk_t0 = time.perf_counter()
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )
                    chunk_elapsed_ms = (time.perf_counter() - chunk_t0) * 1000.0
                    if self.eval_chunk_pause_seconds > 0:
                        time.sleep(self.eval_chunk_pause_seconds)

                    chunk_steps = (
                        int(env_output.dones.shape[1])
                        if env_output.dones is not None and env_output.dones.ndim > 1
                        else int(self.cfg.actor.model.num_action_chunks)
                    )
                    episode_action_chunks += 1
                    episode_steps += chunk_steps
                    total_action_chunks += 1
                    total_env_steps += chunk_steps
                    eval_metrics["baseline_chunk_step_ms"].append(
                        torch.tensor([chunk_elapsed_ms], dtype=torch.float32)
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    episode_done = (
                        env_output.dones is not None and bool(env_output.dones.any())
                    )
                    if episode_done:
                        break

                    is_last_fixed_step = eval_step == self.n_eval_chunk_steps - 1
                    is_last_rollout_epoch = (
                        eval_rollout_epoch == self.cfg.algorithm.eval_rollout_epoch - 1
                    )
                    if not is_last_fixed_step:
                        _send_eval_obs(env_output)
                    elif not is_last_rollout_epoch:
                        extracted_obs, infos = self.eval_env_list[stage_id].reset()
                        env_output = EnvOutput(
                            obs=extracted_obs,
                            final_obs=infos["final_observation"]
                            if "final_observation" in infos
                            else None,
                        )
                        _send_eval_obs(env_output)

                if episode_done:
                    break

            episode_elapsed_s = time.perf_counter() - episode_t0
            total_episode_time_s += episode_elapsed_s
            eval_metrics["baseline_episode_count"].append(
                torch.tensor([1.0], dtype=torch.float32)
            )
            eval_metrics["baseline_action_chunks_per_episode"].append(
                torch.tensor([episode_action_chunks], dtype=torch.float32)
            )
            eval_metrics["baseline_episode_steps"].append(
                torch.tensor([episode_steps], dtype=torch.float32)
            )
            eval_metrics["baseline_episode_time_s"].append(
                torch.tensor([episode_elapsed_s], dtype=torch.float32)
            )
            eval_metrics["baseline_episode_done"].append(
                torch.tensor([float(episode_done)], dtype=torch.float32)
            )

            if not episode_done:
                eval_metrics["success_once"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["return"].append(torch.tensor([0.0], dtype=torch.float32))
                eval_metrics["episode_len"].append(
                    torch.tensor([episode_steps], dtype=torch.float32)
                )
                eval_metrics["reward"].append(torch.tensor([0.0], dtype=torch.float32))
                eval_metrics["intervened_once"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["intervened_steps"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["success_no_intervened"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )

            if episode_done:
                is_last_rollout_epoch = (
                    eval_rollout_epoch == self.cfg.algorithm.eval_rollout_epoch - 1
                )
                stop_all_eval = is_last_rollout_epoch or (
                    stop_eval_on_success and _success_from_info(env_info)
                )
                _send_eval_obs(env_output, eval_stop=stop_all_eval)
                if stop_all_eval:
                    self.finish_rollout(mode="eval")
                    break

            self.finish_rollout(mode="eval")

        eval_metrics["baseline_total_action_chunks"].append(
            torch.tensor([total_action_chunks], dtype=torch.float32)
        )
        eval_metrics["baseline_total_env_steps"].append(
            torch.tensor([total_env_steps], dtype=torch.float32)
        )
        eval_metrics["baseline_total_episode_time_s"].append(
            torch.tensor([total_episode_time_s], dtype=torch.float32)
        )
        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def evaluate_rtc(self, input_channel: Channel, output_channel: Channel):
        self._assert_rtc_eval_supported()
        rtc_cfg = self.cfg.runner.get("rtc", {})
        min_exec_horizon = int(rtc_cfg.get("min_exec_horizon", 2))
        initial_delay_steps = int(rtc_cfg.get("initial_delay_steps", 1))
        delay_buffer_size = int(rtc_cfg.get("delay_buffer_size", 8))

        eval_metrics = defaultdict(list)
        stage_id = 0
        total_episode_steps = 0
        total_episode_time_s = 0.0
        total_deadline_miss = 0
        total_replan_requests = 0
        stop_all_eval = False
        stop_eval_on_success = bool(
            self.cfg.runner.get("stop_eval_on_keyboard_success", False)
        )

        def _success_from_info(env_info: dict[str, Any]) -> bool:
            for key in ("success_once", "success_at_end", "success"):
                value = env_info.get(key)
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    return bool(value.any().item())
                return bool(np.asarray(value).any())
            return False

        def _prepare_rtc_actions(rtc_response: RTCActionResponse):
            chunk_actions = prepare_actions(
                raw_chunk_actions=rtc_response.actions,
                env_type=self.cfg.env.eval.env_type,
                model_type=self.cfg.actor.model.model_type,
                num_action_chunks=self.cfg.actor.model.num_action_chunks,
                action_dim=self.cfg.actor.model.action_dim,
                policy=self.cfg.actor.model.get("policy_setup", None),
                wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            )
            return self._maybe_rewrite_eval_chunk_gripper(chunk_actions)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            self.eval_env_list[stage_id].is_start = True

            extracted_obs, infos = self.eval_env_list[stage_id].reset()
            env_output = EnvOutput(
                obs=extracted_obs,
                final_obs=infos["final_observation"]
                if "final_observation" in infos
                else None,
            )
            env_batch = env_output.to_dict()

            episode_id = eval_rollout_epoch
            chunk_id = 0
            episode_t0 = time.perf_counter()
            episode_step = 0
            deadline_miss = 0
            replan_requests = 0
            episode_success = False
            episode_done = False
            delay_buffer = deque([initial_delay_steps], maxlen=delay_buffer_size)
            eval_metrics["rtc_episode_count"].append(
                torch.tensor([1.0], dtype=torch.float32)
            )

            self.send_rtc_request(
                output_channel,
                RTCRequest(
                    obs=env_batch["obs"],
                    request_type="bootstrap",
                    executed_horizon=0,
                    predicted_delay_steps=initial_delay_steps,
                    chunk_id=chunk_id,
                    episode_id=episode_id,
                ),
                mode="eval",
            )
            rtc_response: RTCActionResponse = self.recv_rtc_response(
                input_channel, mode="eval", async_op=False
            )
            current_chunk_actions = _prepare_rtc_actions(rtc_response)
            current_chunk_index = 0
            current_chunk_len = current_chunk_actions.shape[1]
            pending_rtc_response = None
            request_start_step = 0

            max_eval_steps = self.cfg.env.eval.max_steps_per_rollout_epoch
            while episode_step < max_eval_steps:
                if pending_rtc_response is not None and pending_rtc_response.done():
                    rtc_response = pending_rtc_response.wait()
                    observed_delay_steps = max(episode_step - request_start_step, 0)
                    delay_buffer.append(observed_delay_steps)
                    eval_metrics["rtc_observed_delay_steps"].append(
                        torch.tensor([observed_delay_steps], dtype=torch.float32)
                    )
                    current_chunk_actions = _prepare_rtc_actions(rtc_response)
                    current_chunk_len = current_chunk_actions.shape[1]
                    current_chunk_index = observed_delay_steps
                    pending_rtc_response = None
                    chunk_id = rtc_response.chunk_id

                if (
                    pending_rtc_response is None
                    and current_chunk_index >= min_exec_horizon
                ):
                    predicted_delay_steps = int(max(delay_buffer))
                    self.send_rtc_request(
                        output_channel,
                        RTCRequest(
                            obs=self._copy_rtc_obs(env_output.to_dict()["obs"]),
                            request_type="replan",
                            executed_horizon=current_chunk_index,
                            predicted_delay_steps=predicted_delay_steps,
                            chunk_id=chunk_id + 1,
                            episode_id=episode_id,
                        ),
                        mode="eval",
                    )
                    pending_rtc_response = self.recv_rtc_response(
                        input_channel, mode="eval", async_op=True
                    )
                    request_start_step = episode_step
                    replan_requests += 1

                action_index = current_chunk_index
                if action_index >= current_chunk_len:
                    action_index = current_chunk_len - 1
                    deadline_miss += 1

                env_action = current_chunk_actions[:, action_index]
                env_step_t0 = time.perf_counter()
                env_output, env_info = self.env_evaluate_rtc_action(
                    env_action, stage_id
                )
                eval_metrics["rtc_env_step_ms"].append(
                    torch.tensor(
                        [(time.perf_counter() - env_step_t0) * 1000.0],
                        dtype=torch.float32,
                    )
                )
                if self.eval_chunk_pause_seconds > 0:
                    time.sleep(self.eval_chunk_pause_seconds)

                for key, value in env_info.items():
                    eval_metrics[key].append(value)
                episode_success = episode_success or _success_from_info(env_info)

                episode_step += 1
                current_chunk_index += 1

                episode_done = (
                    env_output.dones is not None and bool(env_output.dones.any())
                )
                if episode_done:
                    break

            episode_elapsed_s = time.perf_counter() - episode_t0
            if pending_rtc_response is not None:
                pending_rtc_response.wait()
                pending_rtc_response = None

            total_episode_steps += episode_step
            total_episode_time_s += episode_elapsed_s
            total_deadline_miss += deadline_miss
            total_replan_requests += replan_requests
            eval_metrics["rtc_deadline_miss_per_episode"].append(
                torch.tensor([deadline_miss], dtype=torch.float32)
            )
            eval_metrics["rtc_replan_requests_per_episode"].append(
                torch.tensor([replan_requests], dtype=torch.float32)
            )
            eval_metrics["rtc_episode_steps"].append(
                torch.tensor([episode_step], dtype=torch.float32)
            )
            eval_metrics["rtc_episode_time_s"].append(
                torch.tensor([episode_elapsed_s], dtype=torch.float32)
            )
            if not episode_done:
                eval_metrics["success_once"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["return"].append(torch.tensor([0.0], dtype=torch.float32))
                eval_metrics["episode_len"].append(
                    torch.tensor([episode_step], dtype=torch.float32)
                )
                eval_metrics["reward"].append(torch.tensor([0.0], dtype=torch.float32))
                eval_metrics["intervened_once"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["intervened_steps"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
                eval_metrics["success_no_intervened"].append(
                    torch.tensor([0.0], dtype=torch.float32)
                )
            self.finish_rollout(mode="eval")
            stop_all_eval = stop_eval_on_success and episode_success
            if stop_all_eval:
                break

        eval_metrics["rtc_total_env_steps"].append(
            torch.tensor([total_episode_steps], dtype=torch.float32)
        )
        eval_metrics["rtc_total_episode_time_s"].append(
            torch.tensor([total_episode_time_s], dtype=torch.float32)
        )
        eval_metrics["rtc_total_deadline_miss"].append(
            torch.tensor([total_deadline_miss], dtype=torch.float32)
        )
        eval_metrics["rtc_total_replan_requests"].append(
            torch.tensor([total_replan_requests], dtype=torch.float32)
        )

        self.send_rtc_request(
            output_channel,
            RTCRequest(
                obs={},
                request_type="stop",
                executed_horizon=0,
                predicted_delay_steps=0,
                chunk_id=0,
                episode_id=-1,
            ),
            mode="eval",
        )

        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(
                self.eval_env_list[stage_id], "offload"
            ):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num
