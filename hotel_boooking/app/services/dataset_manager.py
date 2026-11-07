from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

from app.utils.config import (
    BASE_DATASET_FOLDER,
    UPLOADED_FOLDER,
    MERGED_FOLDER,
    TRAINING_STATE_FILE,
)

logger = logging.getLogger(__name__)


def ensure_dataset_directories() -> None:

	for folder in (
		BASE_DATASET_FOLDER,
		UPLOADED_FOLDER,
		MERGED_FOLDER,
	):
		folder.mkdir(parents=True, exist_ok=True)


def load_training_state() -> dict[str, Any]:

	ensure_dataset_directories()

	if TRAINING_STATE_FILE.exists():

		try:

			return json.loads(TRAINING_STATE_FILE.read_text(encoding="utf-8"))

		except json.JSONDecodeError:
			pass

	return {
		"processed": {},
		"failed": {}
	}


def save_training_state(state: dict[str, Any]) -> None:

	ensure_dataset_directories()

	TRAINING_STATE_FILE.write_text(
		json.dumps(state, indent=2, sort_keys=True),
		encoding="utf-8"
	)


def list_pending_datasets() -> list[Path]:

	ensure_dataset_directories()

	state = load_training_state()

	processed = set(state.get("processed", {}).keys())

	pending_datasets = []

	for dataset_path in sorted(
		UPLOADED_FOLDER.glob("*.csv"),
		key=lambda item: item.stat().st_mtime
	):

		if dataset_path.name not in processed:
			pending_datasets.append(dataset_path)

	return pending_datasets


def mark_dataset_processed(dataset_path: Path | str, training_result: dict[str, Any]) -> None:

	path = Path(dataset_path)

	state = load_training_state()

	state.setdefault("processed", {})[path.name] = {
		"path": str(path),
		"processed_at": datetime.now(timezone.utc).isoformat(),
		"result": training_result
	}

	state.setdefault("failed", {}).pop(path.name, None)

	save_training_state(state)


def mark_dataset_failed(dataset_path: Path | str, error: str) -> None:

	path = Path(dataset_path)

	state = load_training_state()

	state.setdefault("failed", {})[path.name] = {
		"path": str(path),
		"failed_at": datetime.now(timezone.utc).isoformat(),
		"error": error
	}

	save_training_state(state)


def merge_datasets(new_dataset_path: Path | str) -> Path:
	"""
	Concatenates all previously successfully processed datasets with the newly uploaded dataset
	and saves the combined result to MERGED_FOLDER / 'active_dataset.csv'.
	"""
	new_path = Path(new_dataset_path)
	ensure_dataset_directories()

	state = load_training_state()
	processed_files = state.get("processed", {})

	dfs = []
	for name, info in processed_files.items():
		path = Path(info["path"])
		if path.exists():
			try:
				dfs.append(pd.read_csv(path))
			except Exception as e:
				logger.warning(f"Could not read processed dataset {path}: {e}")

	# Append the new dataset
	try:
		dfs.append(pd.read_csv(new_path))
	except Exception as e:
		logger.error(f"Could not read new dataset {new_path}: {e}")
		raise e

	# Merge all
	merged_df = pd.concat(dfs, ignore_index=True)

	# Save to merged folder
	merged_path = MERGED_FOLDER / "active_dataset.csv"
	merged_df.to_csv(merged_path, index=False)

	return merged_path

