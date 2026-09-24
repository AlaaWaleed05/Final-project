"""Tab 2 — Breast Cancer Detection from mammograms (FR-4.1 … FR-4.6)."""
from __future__ import annotations

import time

import streamlit as st

from core import auth, theme
from core.cancer_model import fit_display, get_cancer_service, open_validated_image
from core.config import ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_MB, MIN_IMAGE_SIDE_PX
from core.vitals_model import ModelUnavailable

RESULT_KEY = "cancer_result"
NOT_SET = "Not specified"


def _missing_model(exc: Exception) -> None:
    st.error(f"Breast-cancer model is not available. {exc}")
    st.info("Copy `breast_cancer_model.keras` and `breast_cancer_inference_config.json` from the Tab 2 notebook output into the `models/` folder, then refresh.")


def render() -> None:
    try:
        svc = get_cancer_service()
    except ModelUnavailable as exc:
        _missing_model(exc)
        return

    left, right = st.columns([1, 1.1], gap="large")

    # ------------------------------------------------------------------ input column
    with left:
        with st.container(key="upload_card"):
            theme.section_title("Upload Mammogram (ROI Image)")
            upload = st.file_uploader(
                "Mammogram image", type=list(ALLOWED_IMAGE_EXTENSIONS), key="cancer_upload", label_visibility="collapsed",
                help=f"JPG / PNG, pre-extracted ROI, up to {MAX_UPLOAD_MB} MB",
            )
            st.markdown(f'<div class="hint">JPG / PNG (pre-extracted ROI) · up to {MAX_UPLOAD_MB} MB</div>', unsafe_allow_html=True)

            st.markdown('<div class="hint" style="margin-top:.8rem">Exam details (optional — saved with the result, not used by the model)</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            density = c1.selectbox("Density", [NOT_SET, "A", "B", "C", "D"], key="cancer_density")
            view = c2.selectbox("View", [NOT_SET, "CC", "MLO"], key="cancer_view")
            side = c3.selectbox("Laterality", [NOT_SET, "L", "R"], key="cancer_side")

            analyze = st.button("ANALYZE", type="primary", key="cancer_analyze")

        image, error = None, None
        file_key = None
        if upload is not None:
            file_key = (upload.name, upload.size)
            image, error = open_validated_image(upload.getvalue(), upload.name, ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_MB, MIN_IMAGE_SIDE_PX)
            if error:
                st.error(error)

        if analyze:
            if upload is None:
                st.warning("Upload a mammogram image first.")
            elif image is not None:
                with st.spinner("Analyzing mammogram…"):
                    t0 = time.perf_counter()
                    result = svc.analyze(image)
                    seconds = time.perf_counter() - t0
                st.session_state[RESULT_KEY] = {
                    "result": result, "file_key": file_key, "filename": upload.name, "seconds": seconds,
                    "meta": {"Density": density, "View": view, "Laterality": side},
                }
                auth.log_prediction(auth.current_user()["username"], "mammogram", result.class_name, result.probability)

        stored = st.session_state.get(RESULT_KEY)
        stale = stored is not None and file_key is not None and stored["file_key"] != file_key
        if stale:
            st.info("A new image is loaded — press ANALYZE to update the result.")
        elif stored is not None:
            _render_result_card(stored)

    # ------------------------------------------------------------------ image column
    with right:
        if stored is not None and not stale:
            res = stored["result"]
            show_map = st.toggle("Show Grad-CAM attention map", key="cancer_heat", help="Warm colours = regions that pushed the score most.")
            st.image(res.heatmap_overlay if show_map else res.annotated, width="stretch",
                     caption=f'{stored["filename"]} — ' + ("Grad-CAM attention map" if show_map else "highlighted Point-of-Interest" if res.bbox else "no region highlighted"))
        elif image is not None:
            st.image(fit_display(image.convert("RGB")), width="stretch", caption=f"{upload.name} — preview")
        else:
            st.markdown(
                '<div class="placeholder"><div><div style="font-size:2.4rem">🩻</div>'
                '<b>Your mammogram will appear here</b><br>The highlighted Point-of-Interest is drawn on this image after analysis.</div></div>',
                unsafe_allow_html=True,
            )


def _render_result_card(stored: dict) -> None:
    res = stored["result"]
    lines = [f"Confidence: <b>{res.confidence * 100:.0f}%</b>"]
    lines.append(f"Cancer probability {res.probability:.3f} · decision threshold {res.threshold:.3f}")
    meta = [f"{k}: {v}" for k, v in stored["meta"].items() if v != NOT_SET]
    if meta:
        lines.append(" &nbsp;|&nbsp; ".join(theme.esc(m) for m in meta))
    lines.append(f"Analyzed in {stored['seconds']:.1f} s")

    if res.is_cancer:
        theme.result_card("flag", "Detection Result", "Class: Cancer", "<br>".join(lines))
        if res.bbox is None:
            st.warning("The case is flagged, but the attention is spread out and no single region stood out. Review the whole image.")
    else:
        theme.result_card("clear", "Detection Result", "Class: No Cancer", "<b>No suspicious finding detected.</b><br>" + "<br>".join(lines))

    with st.expander("How to read this result"):
        st.markdown(
            "- **Class** uses a decision threshold tuned on the precision–recall curve (cancer is rare, so 0.5 would miss cases).\n"
            "- **Confidence** is re-centred on that threshold: a score exactly at the threshold reads 50%.\n"
            "- **POI box** is the largest region the network attended to (Grad-CAM). It shows where the model looked, not a confirmed lesion."
        )
    theme.disclaimer()
