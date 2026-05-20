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

import json
import os
import time
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    @staticmethod
    def _jsonable_metric(value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    def _write_eval_summary(self, eval_metrics: dict):
        if not self.cfg.runner.get("write_eval_summary", True):
            return

        log_path = self.cfg.runner.logger.get("log_path", None)
        if not log_path:
            return

        os.makedirs(log_path, exist_ok=True)
        summary_path = os.path.join(log_path, "eval_summary.json")
        summary = {
            key: self._jsonable_metric(value) for key, value in eval_metrics.items()
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Wrote eval summary to {summary_path}")

    def init_workers(self):
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

    def evaluate(self):
        eval_t0 = time.perf_counter()
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        eval_metrics["eval_wall_time_s"] = time.perf_counter() - eval_t0
        eval_metrics["eval_mode_is_rtc"] = 0.0
        return eval_metrics

    def evaluate_rtc(self):
        eval_t0 = time.perf_counter()
        env_handle: Handle = self.env.evaluate_rtc(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate_rtc(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )

        env_results = env_handle.wait()
        rollout_results = rollout_handle.wait()

        env_metrics_list = [results for results in env_results if results is not None]
        rollout_metrics_list = [
            results for results in rollout_results if results is not None
        ]

        env_metrics = compute_evaluate_metrics(env_metrics_list)
        rollout_metrics = compute_evaluate_metrics(rollout_metrics_list)
        rollout_metrics.pop("num_trajectories", None)
        env_metrics.update(rollout_metrics)
        env_metrics["eval_wall_time_s"] = time.perf_counter() - eval_t0
        env_metrics["eval_mode_is_rtc"] = 1.0
        return env_metrics

    def run(self):
        rtc_cfg = self.cfg.runner.get("rtc", {})
        if rtc_cfg.get("enabled", False):
            eval_metrics = self.evaluate_rtc()
        else:
            eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self._write_eval_summary(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
