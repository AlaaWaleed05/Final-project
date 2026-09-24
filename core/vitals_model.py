"""Model Service 1 — vital-signs risk classifier.

Loads the exported scikit-learn / XGBoost pipeline once (cached) and returns a risk level plus
per-prediction feature contributions (SHAP for tree models, occlusion fallback for anything else).

The service reads the column names the pipeline was trained on (``feature_names_in_``) and fills
them from the form. Columns the form cannot provide (e.g. a row index) are passed as NaN, which the
pipeline's own imputer replaces with the training median.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from .clinical import SHORT_NAMES, field_for_column
from .config import HIGH_RISK_CLASS_NAMES, VITALS_LABELS_PATH, VITALS_MODEL_PATH


class ModelUnavailable(RuntimeError):
    """Raised when a model artifact is missing or cannot be loaded (shown as a friendly message)."""


@dataclass
class VitalsPrediction:
    label: str                  # "High Risk" / "Low Risk"
    is_high: bool
    risk_probability: float     # P(high risk)
    contributions: pd.DataFrame  # feature, value, contribution (positive = pushes risk up)
    method: str                 # "SHAP" or "Occlusion"


class VitalsService:
    def __init__(self, pipeline, label_encoder):
        self.pipe = pipeline
        self.classes = [str(c) for c in label_encoder.classes_]
        self.pre = pipeline.named_steps["preprocessor"]
        self.clf = pipeline.named_steps["clf"]

        cols = getattr(self.pre, "feature_names_in_", None)
        if cols is None:
            cols = getattr(pipeline, "feature_names_in_", None)
        if cols is None:
            raise ModelUnavailable("Model pipeline does not expose its training column names.")
        self.columns = [str(c) for c in cols]
        self.column_field = {c: field_for_column(c) for c in self.columns}

        self.high_idx = next((i for i, c in enumerate(self.classes) if c.strip().lower() in HIGH_RISK_CLASS_NAMES), None)
        if self.high_idx is None:
            raise ModelUnavailable(
                f"Could not tell which class means 'high risk' among {self.classes}. "
                "Add the class name to HIGH_RISK_CLASS_NAMES in core/config.py."
            )

    # ---- introspection used by the UI
    @property
    def used_fields(self) -> list[str]:
        seen: list[str] = []
        for f in self.column_field.values():
            if f and f not in seen:
                seen.append(f)
        return seen

    @property
    def unused_fields(self) -> list[str]:
        return [f for f in SHORT_NAMES if f not in self.used_fields]

    # ---- inference
    def _frame(self, values: dict) -> pd.DataFrame:
        row = {}
        for col in self.columns:
            f = self.column_field[col]
            row[col] = values[f] if f else np.nan
        return pd.DataFrame([row], columns=self.columns)

    def _risk_prob(self, frame: pd.DataFrame) -> float:
        return float(self.pipe.predict_proba(frame)[0, self.high_idx])

    def predict(self, values: dict) -> VitalsPrediction:
        frame = self._frame(values)
        p = self._risk_prob(frame)
        is_high = p >= 0.5
        try:
            contrib, method = self._shap(frame), "SHAP"
        except Exception:
            contrib, method = self._occlusion(frame, p), "Occlusion"
        return VitalsPrediction("High Risk" if is_high else "Low Risk", is_high, p, self._to_frame(contrib, values), method)

    # ---- explainability (FR-3.4)
    def _shap(self, frame: pd.DataFrame) -> dict[str, float]:
        import shap

        xt = self.pre.transform(frame)
        if hasattr(xt, "toarray"):
            xt = xt.toarray()
        sv = shap.TreeExplainer(self.clf).shap_values(xt)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        sv = np.asarray(sv)
        if sv.ndim == 3:                       # (rows, features, classes)
            vec = sv[0, :, self.high_idx]
        else:                                  # binary, values refer to class 1
            vec = sv[0] if self.high_idx == 1 else -sv[0]
        names = list(self.pre.get_feature_names_out())
        out: dict[str, float] = {}
        for name, v in zip(names, vec):
            out[self._original_column(name)] = out.get(self._original_column(name), 0.0) + float(v)
        return out

    def _occlusion(self, frame: pd.DataFrame, base_p: float) -> dict[str, float]:
        out = {}
        for col in self.columns:
            if not self.column_field[col]:
                continue
            probe = frame.copy()
            probe[col] = np.nan                # imputer replaces with the training median
            out[col] = base_p - self._risk_prob(probe)
        return out

    def _original_column(self, transformed_name: str) -> str:
        name = transformed_name.split("__", 1)[-1]
        if name in self.columns:
            return name
        for col in self.columns:               # one-hot columns: 'Gender_Male' -> 'Gender'
            if name.startswith(col + "_"):
                return col
        return name

    def _to_frame(self, contrib: dict[str, float], values: dict) -> pd.DataFrame:
        rows = []
        for col, c in contrib.items():
            f = self.column_field.get(col)
            if not f:
                continue                       # not provided by the user -> not explained
            v = values[f]
            rows.append({"feature": SHORT_NAMES[f], "value": v if isinstance(v, str) else float(v), "contribution": float(c)})
        df = pd.DataFrame(rows, columns=["feature", "value", "contribution"])
        df["abs"] = df["contribution"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


@st.cache_resource(show_spinner="Loading vital-signs model…")
def get_vitals_service() -> VitalsService:
    for path in (VITALS_MODEL_PATH, VITALS_LABELS_PATH):
        if not path.exists():
            raise ModelUnavailable(f"Missing file: models/{path.name}")
    try:
        return VitalsService(joblib.load(VITALS_MODEL_PATH), joblib.load(VITALS_LABELS_PATH))
    except ModelUnavailable:
        raise
    except Exception as exc:  # version mismatch, corrupt file, ...
        raise ModelUnavailable(f"Could not load the vital-signs model ({type(exc).__name__}: {exc})") from exc
