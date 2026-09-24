"""Vital-sign field definitions, physiological bounds (FR-3.2) and adult reference ranges.

Hard bounds are the same "plausible physiological range" used in the training notebook, so a value
the app accepts is a value the model could have seen.
"""
from __future__ import annotations

# key -> label, unit, hard (min, max), default, step, normal (low, high)
FIELDS: dict[str, dict] = {
    "hr":   {"label": "Heart Rate",              "unit": "bpm",  "bounds": (20.0, 250.0), "default": 88.0,  "step": 1.0,  "normal": (60.0, 100.0)},
    "resp": {"label": "Respiratory Rate",        "unit": "/min", "bounds": (4.0, 60.0),   "default": 18.0,  "step": 1.0,  "normal": (12.0, 20.0)},
    "temp": {"label": "Body Temperature",        "unit": "°C",   "bounds": (28.0, 43.0),  "default": 37.6,  "step": 0.1,  "normal": (36.1, 37.8)},
    "spo2": {"label": "Oxygen Saturation (SpO₂)", "unit": "%",   "bounds": (40.0, 100.0), "default": 96.0,  "step": 1.0,  "normal": (95.0, 100.0)},
}

# Names shown in explanations (short form)
SHORT_NAMES = {
    "hr": "Heart rate", "resp": "Respiratory rate", "temp": "Body temperature", "spo2": "SpO₂",
}

SAMPLES = {
    "Stable patient": {"hr": 74.0, "resp": 15.0, "temp": 36.8, "spo2": 98.0},
    "Deteriorating patient": {"hr": 128.0, "resp": 29.0, "temp": 39.2, "spo2": 88.0},
}


def field_for_column(column: str) -> str | None:
    key = _guess_field(column)
    return key if key in FIELDS else None   # only fields the form actually has


def _guess_field(column: str) -> str | None:
    """Map a training-set column name (e.g. ' HR (BPM)', 'SpO2 (%)') to an app field key."""
    c = "".join(ch for ch in column.lower() if ch.isalnum())
    if c.startswith("hr") or "heartrate" in c or c.startswith("pulse"):
        return "hr"
    if c.startswith("resp"):
        return "resp"
    if "spo2" in c or "oxygen" in c:
        return "spo2"
    if c.startswith("temp"):
        return "temp"
    if "systolic" in c or c in {"sbp", "bpsys"}:
        return "sbp"
    if "diastolic" in c or c in {"dbp", "bpdia"}:
        return "dbp"
    if c == "age":
        return "age"
    if c in {"gender", "sex"}:
        return "gender"
    return None


def validate_inputs(values: dict) -> list[str]:
    """FR-3.2 — returns human-readable errors (empty list = OK to predict)."""
    errors: list[str] = []
    for key, spec in FIELDS.items():
        v = values.get(key)
        lo, hi = spec["bounds"]
        if v is None:
            errors.append(f"{spec['label']} is required.")
        elif not (lo <= float(v) <= hi):
            errors.append(f"{spec['label']} must be between {lo:g} and {hi:g} {spec['unit']} (got {v:g}).")
    return errors


def reference_check(values: dict) -> list[dict]:
    """Adult reference-range check for every vital (context only — not a model output)."""
    rows = []
    for key, spec in FIELDS.items():
        if spec["normal"] is None:
            continue
        v = float(values[key])
        lo, hi = spec["normal"]
        status = "low" if v < lo else "high" if v > hi else "normal"
        rows.append(
            {
                "key": key,
                "label": SHORT_NAMES[key],
                "value": f"{v:g} {spec['unit']}",
                "range": f"{lo:g}–{hi:g}",
                "status": status,
            }
        )
    return rows
