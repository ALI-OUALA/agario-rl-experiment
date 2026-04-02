"""Background PPO worker coordinator for non-blocking supervisor UI."""

from __future__ import annotations

from queue import Empty, Full, Queue
import threading
from typing import Any

from agario_rl import AgarioConfig
from agario_rl.rl.ppo_shared import SharedPPOTrainer


class AsyncTrainerCoordinator:
    """Handles worker thread lifecycle and non-blocking model updates."""

    def __init__(self, config: AgarioConfig, observation_dim: int) -> None:
        self.config = config
        self.observation_dim = observation_dim
        self.input_queue: Queue[dict[str, Any]] = Queue(maxsize=config.async_training.rollout_queue_size)
        self.output_queue: Queue[dict[str, Any]] = Queue(maxsize=config.async_training.rollout_queue_size + 1)
        self.pending_jobs = 0
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def _worker_loop(self) -> None:
        trainer = SharedPPOTrainer(config=self.config, observation_dim=self.observation_dim, device="cpu")
        while not self._stop_event.is_set():
            try:
                message = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            kind = message.get("kind")
            if kind == "shutdown":
                break
            if kind == "reload":
                trainer.import_training_state(message["state"])
                continue
            if kind != "update":
                continue
            rollout_payload = message["payload"]["rollout"]
            imitation_payload = message["payload"].get("imitation")
            metrics = trainer.update_on_batch(rollout_payload, imitation_payload=imitation_payload)
            self.output_queue.put(
                {
                    "kind": "weights",
                    "state": trainer.export_training_state(),
                    "metrics": metrics,
                }
            )

    def start(self) -> None:
        if not self._worker_thread.is_alive():
            self._worker_thread.start()

    def sync_from_trainer(self, trainer: SharedPPOTrainer) -> None:
        try:
            self.input_queue.put_nowait({"kind": "reload", "state": trainer.export_training_state()})
        except Full:
            pass

    def can_submit(self) -> bool:
        return self.pending_jobs < self.config.async_training.max_pending_weight_updates

    def submit_update(self, payload: dict[str, Any]) -> bool:
        if not self.can_submit():
            return False
        try:
            self.input_queue.put_nowait({"kind": "update", "payload": payload})
        except Full:
            return False
        self.pending_jobs += 1
        return True

    def poll_updates(self, trainer: SharedPPOTrainer) -> dict[str, float] | None:
        latest_metrics: dict[str, float] | None = None
        while True:
            try:
                message = self.output_queue.get_nowait()
            except Empty:
                break
            if message.get("kind") != "weights":
                continue
            trainer.import_training_state(message["state"])
            latest_metrics = dict(message.get("metrics", {}))
            self.pending_jobs = max(0, self.pending_jobs - 1)
        return latest_metrics

    def queue_depth(self) -> int:
        return self.pending_jobs

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self.input_queue.put_nowait({"kind": "shutdown"})
        except Full:
            pass
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
