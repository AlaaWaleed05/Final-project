"""Central paths and constants for the Hospital AI System."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "hospital_ai.db"

# ---- Tab 1 artifacts (exported by the vital-signs notebook) ----
VITALS_MODEL_PATH = MODELS_DIR / "vital_signs_risk_model.joblib"
VITALS_LABELS_PATH = MODELS_DIR / "vital_signs_label_encoder.joblib"

# ---- Tab 2 artifacts (exported by the breast-cancer notebook) ----
CANCER_MODEL_PATH = MODELS_DIR / "breast_cancer_model.keras"
CANCER_CONFIG_PATH = MODELS_DIR / "breast_cancer_inference_config.json"

# Target classes that mean "high risk" (matched case-insensitively against the label encoder).
HIGH_RISK_CLASS_NAMES = {"abnormal", "high risk", "high", "critical"}

# Upload limits (FR-4.2)
ALLOWED_IMAGE_EXTENSIONS = ("png", "jpg", "jpeg")
MAX_UPLOAD_MB = 10
MIN_IMAGE_SIDE_PX = 64
MAX_IMAGE_PIXELS = 60_000_000

# Login throttling
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60

# Palette (shared by CSS and charts)
BLUE = "#2F5BFF"
AQUA = "#00B8A9"
CORAL = "#FF4D6D"
SUN = "#FFB020"
GREEN = "#12B76A"
INK = "#14204B"

DISCLAIMER = (
    "Decision support only — not a certified diagnostic device. "
    "Always refer to the treating physician / radiologist."
)
