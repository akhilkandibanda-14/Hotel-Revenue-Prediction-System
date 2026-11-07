from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from app.services.predictor import RevenuePredictor, revenue_predictor
from app.utils import config

logger = logging.getLogger(__name__)

CAUSAL_DISCLAIMER = (
    "SHAP (SHapley Additive exPlanations) values represent local statistical attributions "
    "within the trained decision-tree ensemble. They measure how each feature pushed the model's "
    "prediction above or below the historical baseline expectation (E[f(X)]). These values do "
    "NOT imply direct counterfactual causality; modifying an operational variable in real-world hotel "
    "operations may yield differing results due to unobserved market dynamics and multi-collinearity."
)


class ExplainerService:
    """
    Explainable AI service utilizing SHAP TreeExplainer.
    Provides global feature importance rankings and single-instance local attributions.
    """

    def __init__(self):
        self._cached_global: Optional[Dict[str, Any]] = None
        self._cached_model_mtime: float = 0.0

    def get_global_importance(self, max_background_samples: int = 60) -> Dict[str, Any]:
        """
        Computes mean absolute SHAP feature importances across a background sample.
        Caches results until the active model file changes on disk.
        """
        revenue_predictor.load_assets()
        model = revenue_predictor.model
        pipeline = revenue_predictor.preprocessor

        if model is None or pipeline is None:
            raise FileNotFoundError("Active model or preprocessing pipeline is not loaded")

        current_mtime = revenue_predictor.model_mtime
        if self._cached_global is not None and self._cached_model_mtime == current_mtime:
            return self._cached_global

        active_path = config.MERGED_FOLDER / "active_dataset.csv"
        if not active_path.exists():
            active_path = config.BASE_DIR / "sample_training_dataset.csv"

        if not active_path.exists():
            raise FileNotFoundError("Background dataset not found for SHAP computation")

        df = pd.read_csv(active_path)
        sample_size = min(len(df), max_background_samples)
        # Use deterministic sampling
        sample_df = df.sample(n=sample_size, random_state=42)

        X_transformed = pipeline.transform(sample_df)
        feature_names = getattr(pipeline, "feature_columns", list(X_transformed.columns))

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_transformed)

        if isinstance(shap_vals, list):
            shap_matrix = np.array(shap_vals[0])
        else:
            shap_matrix = np.array(shap_vals)

        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

        importances = []
        for feat_name, imp in zip(feature_names, mean_abs_shap):
            importances.append({
                "feature": feat_name,
                "importance": float(round(imp, 2)),
            })

        importances = sorted(importances, key=lambda x: x["importance"], reverse=True)

        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(expected_val[0])
        else:
            base_value = float(expected_val)

        result = {
            "model_type": type(model).__name__,
            "background_samples_used": sample_size,
            "base_value": float(round(base_value, 2)),
            "feature_importances": importances,
            "top_features": [f["feature"] for f in importances[:5]],
            "causal_disclaimer": CAUSAL_DISCLAIMER,
        }

        self._cached_global = result
        self._cached_model_mtime = current_mtime
        return result

    def explain_instance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes local SHAP decomposition for an individual prediction request.
        Returns base value, positive contributions, negative contributions, and predicted value.
        """
        revenue_predictor.load_assets()
        model = revenue_predictor.model
        pipeline = revenue_predictor.preprocessor

        if model is None or pipeline is None:
            raise FileNotFoundError("Active model or preprocessing pipeline is not loaded")

        df_input, meta = revenue_predictor.validate_and_prepare_input(input_data, strict=False)
        X_trans = pipeline.transform(df_input)
        feature_names = getattr(pipeline, "feature_columns", list(X_trans.columns))

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            sv_row = np.array(shap_vals[0])[0]
        elif len(shap_vals.shape) > 1:
            sv_row = np.array(shap_vals)[0]
        else:
            sv_row = np.array(shap_vals)

        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_val = float(expected_val[0])
        else:
            base_val = float(expected_val)

        predicted_val = float(model.predict(X_trans)[0])

        positive_contributions: List[Dict[str, Any]] = []
        negative_contributions: List[Dict[str, Any]] = []
        all_contributions: List[Dict[str, Any]] = []

        for name, val, s in zip(feature_names, X_trans.iloc[0], sv_row):
            contrib_val = float(s)
            item = {
                "feature": name,
                "feature_value": float(round(val, 2)) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_value": float(round(contrib_val, 2)),
                "impact": "Increases Revenue" if contrib_val > 0 else "Decreases Revenue",
            }
            all_contributions.append(item)
            if contrib_val > 0:
                positive_contributions.append(item)
            elif contrib_val < 0:
                negative_contributions.append(item)

        positive_contributions = sorted(positive_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        negative_contributions = sorted(negative_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        all_contributions = sorted(all_contributions, key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "predicted_revenue": float(round(predicted_val, 2)),
            "base_value": float(round(base_val, 2)),
            "net_shap_adjustment": float(round(sum(sv_row), 2)),
            "sum_check_matches": bool(abs((base_val + sum(sv_row)) - predicted_val) < 0.1),
            "input_mode": meta.get("mode", "standard"),
            "top_positive_features": positive_contributions[:5],
            "top_negative_features": negative_contributions[:5],
            "all_feature_contributions": all_contributions,
            "causal_disclaimer": CAUSAL_DISCLAIMER,
        }


# Singleton service instance
explainer_service = ExplainerService()

