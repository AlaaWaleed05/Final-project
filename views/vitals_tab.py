"""Tab 1 — Patient Vital-Signs Risk Screening (FR-3.1 … FR-3.4)."""
from __future__ import annotations

import altair as alt
import streamlit as st

from core import auth, theme
from core.clinical import FIELDS, SAMPLES, reference_check, validate_inputs
from core.config import AQUA, CORAL, INK
from core.vitals_model import ModelUnavailable, get_vitals_service

RESULT_KEY = "vitals_result"


def _is_int(key: str) -> bool:
    return FIELDS[key]["step"] == 1.0


def _init_state() -> None:
    for key, spec in FIELDS.items():
        st.session_state.setdefault(f"v_{key}", int(spec["default"]) if _is_int(key) else float(spec["default"]))


def _load_sample(name: str) -> None:
    for key, val in SAMPLES[name].items():
        st.session_state[f"v_{key}"] = int(val) if _is_int(key) else float(val)


def _number(key: str):
    spec = FIELDS[key]
    lo, hi = spec["bounds"]
    if _is_int(key):
        return st.number_input(f'{spec["label"]} ({spec["unit"]})', min_value=int(lo), max_value=int(hi), step=1,
                               key=f"v_{key}", help=f"Valid range: {lo:g}–{hi:g} {spec['unit']}")
    return st.number_input(f'{spec["label"]} ({spec["unit"]})', min_value=float(lo), max_value=float(hi), step=float(spec["step"]),
                           format="%.1f", key=f"v_{key}", help=f"Valid range: {lo:g}–{hi:g} {spec['unit']}")


def _collect() -> dict:
    return {key: float(st.session_state[f"v_{key}"]) for key in FIELDS}


def _missing_model(exc: Exception) -> None:
    st.error(f"Vital-signs model is not available. {exc}")
    st.info("Copy `vital_signs_risk_model.joblib` and `vital_signs_label_encoder.joblib` from the Tab 1 notebook output into the `models/` folder, then refresh.")


def render() -> None:
    try:
        svc = get_vitals_service()
    except ModelUnavailable as exc:
        _missing_model(exc)
        return
    _init_state()

    st.markdown('<div class="hint">Quick demo data:</div>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([1, 1, 4])
    for col, name in zip((b1, b2), SAMPLES):
        col.button(name, key=f"sample_{name}", on_click=_load_sample, args=(name,), width="stretch")

    with st.form("vitals_form"):
        theme.section_title("Patient Vitals")
        left, right = st.columns(2)
        with left:
            _number("hr")
            _number("temp")
        with right:
            _number("resp")
            _number("spo2")
        submitted = st.form_submit_button("PREDICT", type="primary")

    if submitted:
        values = _collect()
        errors = validate_inputs(values)
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            with st.spinner("Analyzing vitals…"):
                pred = svc.predict(values)
            st.session_state[RESULT_KEY] = {"values": values, "pred": pred}
            auth.log_prediction(auth.current_user()["username"], "vitals", pred.label, pred.risk_probability)

    stored = st.session_state.get(RESULT_KEY)
    if stored:
        _render_result(stored, svc)


def _render_result(stored: dict, svc) -> None:
    pred, values = stored["pred"], stored["values"]
    df = pred.contributions

    top = df.head(3)
    factors = ", ".join(
        f'{theme.esc(r.feature)} <i>({"raises" if r.contribution > 0 else "lowers"} risk)</i>' for r in top.itertuples()
    )
    theme.result_card(
        "high" if pred.is_high else "low",
        "Predicted Risk Level",
        pred.label.upper(),
        f"Top contributing factors: {factors}" if factors else "",
    )
    theme.risk_meter(pred.risk_probability)

    if not df.empty:
        st.markdown('<div class="section-title" style="font-size:1.05rem;margin-top:1rem">What pushed this prediction</div>', unsafe_allow_html=True)
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusEnd=6, size=22)
            .encode(
                x=alt.X("contribution:Q", title="← lowers risk      raises risk →", axis=alt.Axis(labelColor=INK, titleColor=INK)),
                y=alt.Y("feature:N", sort=None, title=None, axis=alt.Axis(labelColor=INK, labelFontSize=13)),
                color=alt.condition(alt.datum.contribution > 0, alt.value(CORAL), alt.value(AQUA)),
                tooltip=[alt.Tooltip("feature:N", title="Vital"), alt.Tooltip("value:N", title="Value"), alt.Tooltip("contribution:Q", format="+.3f", title="Effect")],
            )
            .properties(height=max(120, 44 * len(df)))
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, width="stretch")

    st.markdown('<div class="section-title" style="font-size:1.05rem">Reference-range check</div>', unsafe_allow_html=True)
    theme.chips(reference_check(values))

    with st.expander("How this result was produced"):
        used = ", ".join(_pretty(f) for f in svc.used_fields)
        st.markdown(
            f"- **Model inputs:** {used}.\n"
            f"- **Explanation method:** {'SHAP values (exact for tree models)' if pred.method == 'SHAP' else 'occlusion — each vital replaced by its training median and the change in risk measured'}.\n"
            "- **Reference-range check** compares every entered vital with common adult ranges. It is context for the nurse, not a model output."
        )
        unused = [_pretty(f) for f in svc.unused_fields]
        if unused:
            st.caption(f"Entered and validated, but not used by the model (its training data did not include them): {', '.join(unused)}.")
    theme.disclaimer()


def _pretty(field: str) -> str:
    from core.clinical import SHORT_NAMES
    return SHORT_NAMES[field]
