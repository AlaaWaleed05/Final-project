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

from .config import CANCER_CONFIG_PATH, CANCER_MODEL_PATH, MAX_IMAGE_PIXELS
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


def adjusted_confidence(p: float, threshold: float) -> float:
    t = min(max(threshold, 1e-6), 1 - 1e-6)

    if p >= t:
        return 0.5 + 0.5 * (p - t) / (1 - t)

    return 0.5 + 0.5 * (t - p) / t


class CancerService:
    def __init__(self, model, config: dict):
        import tensorflow as tf

        self.tf = tf
        self.model = model

        self.img_size = tuple(
            int(x) for x in config["img_size"]
        )

        self.threshold = float(
            config["decision_threshold"]
        )

        self.class_names = list(
            config.get(
                "class_names",
                ["No Cancer", "Cancer"]
            )
        )

        self.last_conv = config[
            "last_conv_layer_name"
        ]

        backbone = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model)
            ),
            None,
        )

        if backbone is None:
            backbone = next(
                (
                    layer
                    for layer in model.layers
                    if "efficientnet" in layer.name.lower()
                ),
                None,
            )

        if backbone is None:
            raise ModelUnavailable(
                "Could not find the EfficientNet backbone "
                "inside the saved model."
            )

        self.conv_model = tf.keras.Model(
            backbone.input,
            [
                backbone.get_layer(
                    self.last_conv
                ).output,
                backbone.output,
            ],
        )

        self.head_layers = model.layers[
            model.layers.index(backbone) + 1:
        ]

    def _forward(
        self,
        rgb_small: np.ndarray
    ) -> tuple[float, np.ndarray]:

        tf = self.tf

        batch = tf.convert_to_tensor(
            rgb_small[None].astype("float32")
        )

        with tf.GradientTape() as tape:
            conv_out, feats = self.conv_model(
                batch,
                training=False
            )

            x = feats

            for layer in self.head_layers:
                x = layer(
                    x,
                    training=False
                )

            score = x[:, 0]

        grads = tape.gradient(
            score,
            conv_out
        )

        pooled = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        cam = tf.squeeze(
            conv_out[0] @ pooled[..., tf.newaxis]
        )

        cam = tf.maximum(cam, 0) / (
            tf.math.reduce_max(cam) + 1e-9
        )

        return (
            float(score.numpy()[0]),
            cam.numpy()
        )

    def analyze(
        self,
        image: Image.Image
    ) -> CancerResult:

        rgb = image.convert("RGB")

        small = rgb.resize(
            self.img_size,
            Image.NEAREST
        )

        prob, cam = self._forward(
            np.asarray(small)
        )

        display = fit_display(rgb)

        h, w = display.shape[:2]

        cam_big = cv2.resize(
            cam,
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        cam_big = np.clip(
            cam_big,
            0,
            1
        )

        is_cancer = (
            prob >= self.threshold
        )

        bbox = (
            _largest_region(cam_big)
            if is_cancer
            else None
        )

        annotated = display.copy()

        if bbox is not None:
            _draw_poi(
                annotated,
                bbox
            )

        return CancerResult(
            probability=prob,
            threshold=self.threshold,
            is_cancer=is_cancer,
            confidence=adjusted_confidence(
                prob,
                self.threshold
            ),
            bbox=bbox,
            original=display,
            annotated=annotated,
            heatmap_overlay=_overlay(
                display,
                cam_big
            ),
            class_name=(
                self.class_names[1]
                if is_cancer
                else self.class_names[0]
            ),
        )


def fit_display(
    rgb: Image.Image
) -> np.ndarray:

    w, h = rgb.size

    scale = min(
        1.0,
        DISPLAY_MAX_SIDE / max(w, h)
    )

    if scale < 1.0:
        rgb = rgb.resize(
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
            Image.LANCZOS
        )

    return np.asarray(
        rgb,
        dtype=np.uint8
    ).copy()


def _largest_region(
    cam: np.ndarray
) -> tuple[int, int, int, int] | None:

    if cam.max() < 0.05:
        return None

    mask = (
        cam >= POI_RELATIVE_THRESHOLD * cam.max()
    ).astype("uint8")

    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if n <= 1:
        return None

    idx = 1 + int(
        np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
    )

    x, y, w, h, area = stats[idx]

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
        x - pad_x
    )

    y0 = max(
        0,
        y - pad_y
    )

    x1 = min(
        cam.shape[1],
        x + w + pad_x
    )

    y1 = min(
        cam.shape[0],
        y + h + pad_y
    )

    return (
        int(x0),
        int(y0),
        int(x1 - x0),
        int(y1 - y0),
    )


def _draw_poi(
    img: np.ndarray,
    bbox: tuple[int, int, int, int]
) -> None:

    x, y, w, h = bbox

    thick = max(
        2,
        int(
            round(
                max(img.shape[:2]) / 250
            )
        )
    )

    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        POI_RED,
        thick
    )

    font_scale = max(
        0.5,
        max(img.shape[:2]) / 1400
    )

    label = "POI"

    (tw, th), base = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        max(1, thick - 1)
    )

    top = max(
        0,
        y - th - base - 6
    )

    cv2.rectangle(
        img,
        (x, top),
        (
            x + tw + 12,
            top + th + base + 6
        ),
        POI_RED,
        -1
    )

    cv2.putText(
        img,
        label,
        (x + 6, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, thick - 1),
        cv2.LINE_AA
    )


def _overlay(
    display: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:

    colour = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    colour = cv2.cvtColor(
        colour,
        cv2.COLOR_BGR2RGB
    )

    return np.clip(
        (1 - alpha) * display
        + alpha * colour,
        0,
        255
    ).astype(np.uint8)


def open_validated_image(
    data: bytes,
    filename: str,
    allowed_ext: tuple[str, ...],
    max_mb: int,
    min_side: int,
) -> tuple[
    Image.Image | None,
    str | None
]:

    ext = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if ext not in allowed_ext:
        return (
            None,
            f"Unsupported file type '.{ext}'. "
            f"Upload a "
            f"{', '.join(e.upper() for e in allowed_ext)} "
            f"image."
        )

    if len(data) > max_mb * 1024 * 1024:
        return (
            None,
            f"File is larger than {max_mb} MB."
        )

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

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
            "Check that it is a valid PNG or JPG."
        )

    if img.format not in (
        "PNG",
        "JPEG"
    ):
        return (
            None,
            f"The file content is {img.format}, "
            "not PNG/JPG."
        )

    if min(img.size) < min_side:
        return (
            None,
            f"Image is too small "
            f"({img.size[0]}×{img.size[1]} px). "
            f"Minimum side is {min_side} px."
        )

    return img, None


@st.cache_resource(
    show_spinner="Loading breast-cancer model…"
)
def get_cancer_service() -> CancerService:

    # Download the large Keras model from Google Drive
    # if it is not already available locally.
    if not CANCER_MODEL_PATH.exists():

        CANCER_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        file_id = (
            "1eVBfhoeiXeoR4PmpOsadQ6p6hoofyJ2u"
        )

        url = (
            f"https://drive.google.com/uc?id={file_id}"
        )

        downloaded = gdown.download(
            url,
            str(CANCER_MODEL_PATH),
            quiet=False,
        )

        if downloaded is None:
            raise ModelUnavailable(
                "Could not download the breast-cancer model "
                "from Google Drive."
            )

    if not CANCER_CONFIG_PATH.exists():
        raise ModelUnavailable(
            f"Missing file: "
            f"{CANCER_CONFIG_PATH.name}"
        )

    try:

        import tensorflow as tf

        model = tf.keras.models.load_model(
            CANCER_MODEL_PATH,
            compile=False,
        )

        config = json.loads(
            CANCER_CONFIG_PATH.read_text()
        )

        return CancerService(
            model,
            config
        )

    except ModelUnavailable:
        raise

    except Exception as exc:
        raise ModelUnavailable(
            "Could not load the breast-cancer model "
            f"({type(exc).__name__}: {exc})"
        ) from exc
