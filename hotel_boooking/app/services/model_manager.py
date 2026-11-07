from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.utils import config

logger = logging.getLogger(__name__)


def ensure_model_directories() -> None:
    config.get_model_folder().mkdir(parents=True, exist_ok=True)
    config.get_versions_folder().mkdir(parents=True, exist_ok=True)


def get_next_version() -> str:
    ensure_model_directories()
    existing = list(config.get_versions_folder().glob("model_v*.joblib"))
    if not existing:
        return "v001"
    
    versions = []
    for p in existing:
        stem = p.stem.replace("model_v", "")
        if stem.isdigit():
            versions.append(int(stem))
    
    next_num = max(versions, default=0) + 1
    return f"v{next_num:03d}"


def list_available_versions() -> List[str]:
    """Returns list of all stored version identifiers, sorted newest first."""
    ensure_model_directories()
    versions = []
    for p in sorted(config.get_versions_folder().glob("model_v*.joblib"), reverse=True):
        stem = p.stem.replace("model_", "")
        versions.append(stem)
    return versions


def save_atomic_joblib(obj: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=str(destination.parent))
    temp_file = Path(temp_dir) / destination.name

    try:
        joblib.dump(obj, temp_file)
        # Verify readability before replacing destination
        _ = joblib.load(temp_file)
        shutil.move(str(temp_file), str(destination))
    finally:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def save_atomic_json(data: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=str(destination.parent))
    temp_file = Path(temp_dir) / destination.name

    try:
        temp_file.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        shutil.move(str(temp_file), str(destination))
    finally:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_model_metadata() -> Dict[str, Any]:
    meta_path = config.get_metadata_path()
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read metadata file: {e}")
    return {}


def load_model_comparison() -> Dict[str, Any]:
    comp_path = config.get_comparison_path()
    if comp_path.exists():
        try:
            return json.loads(comp_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read comparison file: {e}")
    return {}


def deploy_trained_model(
    model: Any,
    preprocessor: Any,
    selected_algorithm: str,
    comparison_results: List[Dict[str, Any]],
    best_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Saves and promotes a validated best model to production atomically.
    Preserves version history and legacy pickle formats.
    """
    ensure_model_directories()
    version = get_next_version()
    timestamp = datetime.now(timezone.utc).isoformat()

    # 1. Save versioned model artifact
    version_artifact_path = config.get_versions_folder() / f"model_{version}.joblib"
    save_atomic_joblib(model, version_artifact_path)
    logger.info(f"Saved versioned model artifact: {version_artifact_path.name}")

    # 2. Save current production model atomically
    save_atomic_joblib(model, config.get_current_model_path())
    # Also save to legacy MODEL_PATH (.pkl) for full backward compatibility
    save_atomic_joblib(model, config.get_model_path())

    # 3. Save preprocessor pipeline
    save_atomic_joblib(preprocessor, config.get_pipeline_path())

    # 4. Save legacy feature_columns and label_encoders for backward compatibility
    if hasattr(preprocessor, "feature_columns"):
        save_atomic_joblib(preprocessor.feature_columns, config.get_feature_path())

    legacy_encoders = {}
    if hasattr(preprocessor, "encoder") and preprocessor.encoder is not None:
        cat_cols = [c for c in preprocessor.categorical_cols if c in preprocessor.feature_columns]
        for idx, col in enumerate(cat_cols):
            classes = list(preprocessor.encoder.categories_[idx])
            le = LabelEncoder()
            le.classes_ = np.array(classes)
            legacy_encoders[col] = le
    save_atomic_joblib(legacy_encoders, config.get_encoder_path())

    # 5. Save model metadata
    metadata = {
        "version": version,
        "algorithm": selected_algorithm,
        "deployed_at": timestamp,
        "artifact_path": str(version_artifact_path),
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "dataset_info": dataset_info,
        "feature_count": len(preprocessor.feature_columns) if hasattr(preprocessor, "feature_columns") else 0,
        "feature_columns": getattr(preprocessor, "feature_columns", []),
        "status": "production",
    }
    save_atomic_json(metadata, config.get_metadata_path())

    # 6. Save model comparison results
    comparison_payload = {
        "timestamp": timestamp,
        "winner": selected_algorithm,
        "version": version,
        "models": comparison_results,
    }
    save_atomic_json(comparison_payload, config.get_comparison_path())

    logger.info(f"Successfully deployed production model {version} ({selected_algorithm}).")
    return metadata


def rollback_to_version(version: str) -> Dict[str, Any]:
    """
    Rolls back the active production model to a previously saved version.
    """
    ensure_model_directories()
    target_path = config.get_versions_folder() / f"model_{version}.joblib"
    if not target_path.exists():
        raise FileNotFoundError(f"Target model version '{version}' not found in {config.get_versions_folder()}")

    # Load target to verify integrity
    model = joblib.load(target_path)

    # Atomically replace production models
    save_atomic_joblib(model, config.get_current_model_path())
    save_atomic_joblib(model, config.get_model_path())

    # Update metadata
    current_meta = load_model_metadata()
    current_meta["version"] = version
    current_meta["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    current_meta["artifact_path"] = str(target_path)
    save_atomic_json(current_meta, config.get_metadata_path())

    logger.info(f"Production model successfully rolled back to version {version}.")
    return current_meta
