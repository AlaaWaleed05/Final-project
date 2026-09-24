"""Model Service 2 — mammogram Cancer / No Cancer classifier + Grad-CAM Point-of-Interest."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import gdown
import numpy as np
import streamlit as st
from PIL import Image

from .config import (
    CANCER_CONFIG_PATH,
    CANCER_MODEL_PATH,
    MAX_IMAGE_PIXELS,
)
from .vitals_model import ModelUnavailable


DISPLAY_MAX_SIDE = 1000
POI_RELATIVE_THRESHOLD = 0.60
POI_MIN_AREA_FRACTION = 0.002
POI_RED = (255, 45, 85)


@dataclass
class CancerResult:
    probability: float
    threshold: float
    is_cancer: bool
    confidence: float
    bbox: tuple[int, int, int, int] | None
    original: np.ndarray
    annotated: np.ndarray
    heatmap_overlay: np.ndarray
    class_name: str


def adjusted_confidence(
    p: float,
    threshold: float,
) -> float:

    t = min(
        max(threshold, 1e-6),
        1 - 1e-6,
    )

    if p >= t:
        return (
            0.5
            + 0.5 * (p - t) / (1 - t)
        )

    return (
        0.5
        + 0.5 * (t - p) / t
    )


class CancerService:

    def __init__(
        self,
        model,
        config: dict,
    ):

        import tensorflow as tf

        self.tf = tf
        self.model = model

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.img_size = tuple(
            int(x)
            for x in config["img_size"]
        )

        self.threshold = float(
            config["decision_threshold"]
        )

        self.class_names = list(
            config.get(
                "class_names",
                [
                    "No Cancer",
                    "Cancer",
                ],
            )
        )

        self.last_conv = config[
            "last_conv_layer_name"
        ]

        # ---------------------------------------------------------
        # Find EfficientNet backbone
        # ---------------------------------------------------------

        backbone = next(
            (
                layer
                for layer in model.layers
                if isinstance(
                    layer,
                    tf.keras.Model,
                )
            ),
            None,
        )

        if backbone is None:

            backbone = next(
                (
                    layer
                    for layer in model.layers
                    if "efficientnet"
                    in layer.name.lower()
                ),
                None,
            )

        if backbone is None:

            raise ModelUnavailable(
                "Could not find the EfficientNet "
                "backbone inside the saved model."
            )

        self.backbone = backbone

        # ---------------------------------------------------------
        # Find Grad-CAM layer
        # ---------------------------------------------------------

        try:

            last_conv_layer = (
                backbone.get_layer(
                    self.last_conv
                )
            )

        except Exception as exc:

            raise ModelUnavailable(
                "Could not find Grad-CAM layer "
                f"'{self.last_conv}' inside "
                "the backbone."
            ) from exc

        self.last_conv_layer = (
            last_conv_layer
        )

        # ---------------------------------------------------------
        # Grad-CAM model
        # ---------------------------------------------------------

        self.conv_model = tf.keras.Model(
            backbone.input,
            [
                last_conv_layer.output,
                backbone.output,
            ],
        )

        # ---------------------------------------------------------
        # Layers after backbone
        # ---------------------------------------------------------

        backbone_index = (
            model.layers.index(backbone)
        )

        self.head_layers = model.layers[
            backbone_index + 1:
        ]

        # ---------------------------------------------------------
        # IMPORTANT
        #
        # We don't assume that backbone.output is a Tensor.
        # It can be a Tensor, list, or tuple.
        #
        # The current model previously returned (1, 7, 1),
        # so we need to inspect the actual architecture before
        # deciding how to convert it into a cancer probability.
        # ---------------------------------------------------------

        self._run_model_diagnostic()

    # =============================================================
    # MODEL DIAGNOSTIC
    # =============================================================

    def _describe_value(
        self,
        value,
        name: str,
    ) -> list[str]:

        tf = self.tf

        if isinstance(
            value,
            (list, tuple),
        ):

            lines = [
                f"{name}: "
                f"{type(value).__name__} "
                f"(length={len(value)})"
            ]

            for index, item in enumerate(value):

                lines.extend(
                    self._describe_value(
                        item,
                        f"{name}[{index}]",
                    )
                )

            return lines

        if isinstance(
            value,
            dict,
        ):

            lines = [
                f"{name}: dict "
                f"(keys={list(value.keys())})"
            ]

            for key, item in value.items():

                lines.extend(
                    self._describe_value(
                        item,
                        f"{name}[{key!r}]",
                    )
                )

            return lines

        try:

            shape = value.shape

        except Exception:

            shape = None

        try:

            dtype = value.dtype

        except Exception:

            dtype = None

        if isinstance(
            value,
            tf.Tensor,
        ):

            value_type = "Tensor"

        else:

            value_type = type(value).__name__

        return [
            f"{name}: "
            f"type={value_type}, "
            f"shape={shape}, "
            f"dtype={dtype}"
        ]

    def _run_model_diagnostic(self):

        tf = self.tf

        try:

            dummy = tf.zeros(
                (
                    1,
                    self.img_size[0],
                    self.img_size[1],
                    3,
                ),
                dtype=tf.float32,
            )

            conv_result = (
                self.conv_model(
                    dummy,
                    training=False,
                )
            )

            if not isinstance(
                conv_result,
                (list, tuple),
            ):

                raise ModelUnavailable(
                    "Unexpected Grad-CAM model output: "
                    f"{type(conv_result).__name__}"
                )

            if len(conv_result) < 2:

                raise ModelUnavailable(
                    "Grad-CAM model returned fewer "
                    "than two outputs."
                )

            conv_out = conv_result[0]
            backbone_features = conv_result[1]

            shape_trace = []

            shape_trace.append(
                "===== CANCER MODEL DIAGNOSTIC ====="
            )

            shape_trace.extend(
                self._describe_value(
                    conv_out,
                    "Grad-CAM output",
                )
            )

            shape_trace.extend(
                self._describe_value(
                    backbone_features,
                    "Backbone output",
                )
            )

            # -----------------------------------------------------
            # If backbone output is a list/tuple, inspect branches
            # -----------------------------------------------------

            if isinstance(
                backbone_features,
                (list, tuple),
            ):

                for branch_index, branch in enumerate(
                    backbone_features
                ):

                    shape_trace.append("")
                    shape_trace.append(
                        f"===== BRANCH {branch_index} ====="
                    )

                    x = branch

                    for layer in self.head_layers:

                        try:

                            x = layer(
                                x,
                                training=False,
                            )

                        except Exception as exc:

                            shape_trace.append(
                                f"{layer.name} "
                                f"({layer.__class__.__name__}) "
                                f"FAILED: "
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            )

                            break

                        shape_trace.extend(
                            self._describe_value(
                                x,
                                layer.name,
                            )
                        )

            else:

                # -------------------------------------------------
                # Normal Tensor output
                # -------------------------------------------------

                x = backbone_features

                shape_trace.append("")
                shape_trace.append(
                    "===== CLASSIFIER HEAD ====="
                )

                for layer in self.head_layers:

                    try:

                        x = layer(
                            x,
                            training=False,
                        )

                    except Exception as exc:

                        shape_trace.append(
                            f"{layer.name} "
                            f"({layer.__class__.__name__}) "
                            f"FAILED: "
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                        break

                    shape_trace.extend(
                        self._describe_value(
                            x,
                            layer.name,
                        )
                    )

            shape_trace.append("")
            shape_trace.append(
                "===== END DIAGNOSTIC ====="
            )

            # -----------------------------------------------------
            # Stop intentionally so we don't make an incorrect
            # medical classification.
            # -----------------------------------------------------

            raise ModelUnavailable(
                "\n".join(shape_trace)
            )

        except ModelUnavailable:
            raise

        except Exception as exc:

            raise ModelUnavailable(
                "Could not inspect cancer model output: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # =============================================================
    # FORWARD PASS
    # =============================================================

    def _forward(
        self,
        rgb_small: np.ndarray,
    ) -> tuple[float, np.ndarray]:

        tf = self.tf

        batch = tf.convert_to_tensor(
            rgb_small[None].astype(
                "float32"
            )
        )

        with tf.GradientTape() as tape:

            conv_out, feats = (
                self.conv_model(
                    batch,
                    training=False,
                )
            )

            # -----------------------------------------------------
            # The final model architecture has not yet been
            # identified. Do not guess how to reduce multiple
            # outputs into one cancer probability.
            # -----------------------------------------------------

            if isinstance(
                feats,
                (list, tuple),
            ):

                raise ModelUnavailable(
                    "The EfficientNet backbone returns "
                    "multiple outputs. The classifier "
                    "head mapping has not been determined yet."
                )

            x = feats

            for layer in self.head_layers:

                x = layer(
                    x,
                    training=False,
                )

            score_values = tf.reshape(
                x,
                [-1],
            )

            if int(
                tf.size(score_values).numpy()
            ) != 1:

                raise ModelUnavailable(
                    "Unexpected classifier output shape: "
                    f"{x.shape}. "
                    "The model does not return exactly "
                    "one cancer probability."
                )

            score = score_values[0]

        grads = tape.gradient(
            score,
            conv_out,
        )

        if grads is None:

            raise ModelUnavailable(
                "Could not calculate Grad-CAM "
                "gradients for the breast-cancer model."
            )

        pooled = tf.reduce_mean(
            grads,
            axis=(0, 1, 2),
        )

        cam = tf.squeeze(
            conv_out[0]
            @ pooled[..., tf.newaxis]
        )

        cam = tf.maximum(
            cam,
            0,
        )

        cam = cam / (
            tf.math.reduce_max(cam)
            + 1e-9
        )

        probability = float(
            score.numpy()
        )

        return (
            probability,
            cam.numpy(),
        )

    # =============================================================
    # ANALYZE IMAGE
    # =============================================================

    def analyze(
        self,
        image: Image.Image,
    ) -> CancerResult:

        rgb = image.convert(
            "RGB"
        )

        small = rgb.resize(
            self.img_size,
            Image.NEAREST,
        )

        prob, cam = self._forward(
            np.asarray(small)
        )

        display = fit_display(
            rgb
        )

        h, w = display.shape[:2]

        cam_big = cv2.resize(
            cam,
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )

        cam_big = np.clip(
            cam_big,
            0,
            1,
        )

        is_cancer = (
            prob >= self.threshold
        )

        bbox = (
            _largest_region(
                cam_big
            )
            if is_cancer
            else None
        )

        annotated = display.copy()

        if bbox is not None:

            _draw_poi(
                annotated,
                bbox,
            )

        return CancerResult(
            probability=prob,
            threshold=self.threshold,
            is_cancer=is_cancer,
            confidence=adjusted_confidence(
                prob,
                self.threshold,
            ),
            bbox=bbox,
            original=display,
            annotated=annotated,
            heatmap_overlay=_overlay(
                display,
                cam_big,
            ),
            class_name=(
                self.class_names[1]
                if is_cancer
                else self.class_names[0]
            ),
        )


# =============================================================
# DISPLAY
# =============================================================

def fit_display(
    rgb: Image.Image,
) -> np.ndarray:

    w, h = rgb.size

    scale = min(
        1.0,
        DISPLAY_MAX_SIDE
        / max(w, h),
    )

    if scale < 1.0:

        rgb = rgb.resize(
            (
                max(
                    1,
                    int(w * scale),
                ),
                max(
                    1,
                    int(h * scale),
                ),
            ),
            Image.LANCZOS,
        )

    return np.asarray(
        rgb,
        dtype=np.uint8,
    ).copy()


# =============================================================
# FIND LARGEST CAM REGION
# =============================================================

def _largest_region(
    cam: np.ndarray,
) -> tuple[
    int,
    int,
    int,
    int,
] | None:

    if cam.max() < 0.05:
        return None

    mask = (
        cam
        >= POI_RELATIVE_THRESHOLD
        * cam.max()
    ).astype("uint8")

    n, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    )

    if n <= 1:
        return None

    idx = (
        1
        + int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )
        )
    )

    x, y, w, h, area = stats[
        idx
    ]

    if area < (
        POI_MIN_AREA_FRACTION
        * cam.shape[0]
        * cam.shape[1]
    ):

        return None

    pad_x = int(
        0.03 * cam.shape[1]
    )

    pad_y = int(
        0.03 * cam.shape[0]
    )

    x0 = max(
        0,
        x - pad_x,
    )

    y0 = max(
        0,
        y - pad_y,
    )

    x1 = min(
        cam.shape[1],
        x + w + pad_x,
    )

    y1 = min(
        cam.shape[0],
        y + h + pad_y,
    )

    return (
        int(x0),
        int(y0),
        int(x1 - x0),
        int(y1 - y0),
    )


# =============================================================
# DRAW POINT OF INTEREST
# =============================================================

def _draw_poi(
    img: np.ndarray,
    bbox: tuple[
        int,
        int,
        int,
        int,
    ],
) -> None:

    x, y, w, h = bbox

    thick = max(
        2,
        int(
            round(
                max(
                    img.shape[:2]
                )
                / 250
            )
        ),
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        POI_RED,
        thick,
    )

    font_scale = max(
        0.5,
        max(
            img.shape[:2]
        )
        / 1400,
    )

    label = "POI"

    (
        tw,
        th,
    ), base = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        max(
            1,
            thick - 1,
        ),
    )

    top = max(
        0,
        y - th - base - 6,
    )

    cv2.rectangle(
        img,
        (x, top),
        (
            x + tw + 12,
            top + th + base + 6,
        ),
        POI_RED,
        -1,
    )

    cv2.putText(
        img,
        label,
        (
            x + 6,
            top + th + 2,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(
            1,
            thick - 1,
        ),
        cv2.LINE_AA,
    )


# =============================================================
# HEATMAP OVERLAY
# =============================================================

def _overlay(
    display: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:

    colour = cv2.applyColorMap(
        np.uint8(
            255 * cam
        ),
        cv2.COLORMAP_JET,
    )

    colour = cv2.cvtColor(
        colour,
        cv2.COLOR_BGR2RGB,
    )

    return np.clip(
        (1 - alpha) * display
        + alpha * colour,
        0,
        255,
    ).astype(np.uint8)


# =============================================================
# IMAGE VALIDATION
# =============================================================

def open_validated_image(
    data: bytes,
    filename: str,
    allowed_ext: tuple[str, ...],
    max_mb: int,
    min_side: int,
) -> tuple[
    Image.Image | None,
    str | None,
]:

    ext = (
        filename.rsplit(
            ".",
            1,
        )[-1].lower()
        if "." in filename
        else ""
    )

    if ext not in allowed_ext:

        return (
            None,
            f"Unsupported file type '.{ext}'. "
            f"Upload a "
            f"{', '.join(e.upper() for e in allowed_ext)} "
            f"image.",
        )

    if len(data) > (
        max_mb * 1024 * 1024
    ):

        return (
            None,
            f"File is larger than {max_mb} MB.",
        )

    Image.MAX_IMAGE_PIXELS = (
        MAX_IMAGE_PIXELS
    )

    try:

        probe = Image.open(
            io.BytesIO(data)
        )

        probe.verify()

        img = Image.open(
            io.BytesIO(data)
        )

        img.load()

    except Exception:

        return (
            None,
            "This file is not a readable image. "
            "Check that it is a valid PNG or JPG.",
        )

    if img.format not in (
        "PNG",
        "JPEG",
    ):

        return (
            None,
            f"The file content is {img.format}, "
            "not PNG/JPG.",
        )

    if min(img.size) < min_side:

        return (
            None,
            f"Image is too small "
            f"({img.size[0]}×{img.size[1]} px). "
            f"Minimum side is {min_side} px.",
        )

    return img, None


# =============================================================
# LOAD MODEL
# =============================================================

@st.cache_resource(
    show_spinner="Loading breast-cancer model…"
)
def get_cancer_service() -> CancerService:

    # ---------------------------------------------------------
    # Download model if necessary
    # ---------------------------------------------------------

    if not CANCER_MODEL_PATH.exists():

        CANCER_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_id = (
            "1eVBfhoeiXeoR4PmpOsadQ6p6hoofyJ2u"
        )

        url = (
            "https://drive.google.com/uc"
            f"?id={file_id}"
        )

        downloaded = gdown.download(
            url,
            str(CANCER_MODEL_PATH),
            quiet=False,
        )

        if downloaded is None:

            raise ModelUnavailable(
                "Could not download the "
                "breast-cancer model "
                "from Google Drive."
            )

    # ---------------------------------------------------------
    # Config
    # ---------------------------------------------------------

    if not CANCER_CONFIG_PATH.exists():

        raise ModelUnavailable(
            f"Missing file: "
            f"{CANCER_CONFIG_PATH.name}"
        )

    try:

        import tensorflow as tf

        model = (
            tf.keras.models.load_model(
                CANCER_MODEL_PATH,
                compile=False,
            )
        )

        config = json.loads(
            CANCER_CONFIG_PATH.read_text()
        )

        return CancerService(
            model,
            config,
        )

    except ModelUnavailable:
        raise

    except Exception as exc:

        raise ModelUnavailable(
            "Could not load the breast-cancer "
            "model "
            f"({type(exc).__name__}: {exc})"
        ) from exc
