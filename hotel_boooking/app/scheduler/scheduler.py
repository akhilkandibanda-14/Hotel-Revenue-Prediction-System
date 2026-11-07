from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from app.services.dataset_manager import (
	list_pending_datasets,
	mark_dataset_failed,
	mark_dataset_processed,
	merge_datasets,
)
from app.services.trainer import train_model
from app.utils.config import POLL_INTERVAL_SECONDS


logger = logging.getLogger(__name__)


@dataclass
class TrainingScheduler:

	poll_interval_seconds: int = POLL_INTERVAL_SECONDS
	_stop_event: threading.Event = field(default_factory=threading.Event, init=False)
	_thread: threading.Thread | None = field(default=None, init=False)

	def start(self) -> None:

		if self._thread and self._thread.is_alive():
			return

		self._stop_event.clear()

		self._thread = threading.Thread(
			target=self._run,
			name="training-scheduler",
			daemon=True,
		)

		self._thread.start()

	def stop(self) -> None:

		self._stop_event.set()

		if self._thread and self._thread.is_alive():
			self._thread.join(timeout=5)

	def _run(self) -> None:

		logger.info("Training scheduler started")

		while not self._stop_event.is_set():

			pending_datasets = list_pending_datasets()

			if not pending_datasets:
				self._stop_event.wait(self.poll_interval_seconds)
				continue

			for dataset_path in pending_datasets:

				if self._stop_event.is_set():
					break

				try:
					logger.info("Merging and training dataset %s", dataset_path)
					merged_path = merge_datasets(dataset_path)
					result = train_model(str(merged_path))
					mark_dataset_processed(dataset_path, result)

				except Exception as exc:
					logger.exception("Training failed for %s", dataset_path)
					mark_dataset_failed(dataset_path, str(exc))

		logger.info("Training scheduler stopped")


training_scheduler = TrainingScheduler()
