# Hospital Multi-Model AI Risk Screening & Breast Cancer Detection System

Streamlit app for the graduation project PRD: one login, two independent tabs
(Vital-Signs Risk Screening, Breast Cancer Detection).

## Run

```bash
pip install -r requirements.txt
# copy the 4 exported model files into models/  (see models/README.md)
streamlit run app.py
```

Demo accounts are created on first run (hashed in `data/hospital_ai.db`):

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin@2026` | System Admin (sees the Admin panel) |
| `staff_sara` | `Sara@2026` | Ward nurse |
| `staff_omar` | `Omar@2026` | Radiology technician |

Change or deactivate them from the Admin panel before any real deployment.

## Layout

```
app.py                 entry point: login gate → header → two tabs
core/auth.py           login, PBKDF2 hashing, session, SQLite users + prediction log
core/vitals_model.py   Model Service 1 (cached): predict + SHAP explanation
core/cancer_model.py   Model Service 2 (cached): classify + Grad-CAM POI box + upload validation
core/clinical.py       field definitions, physiological bounds, reference ranges
core/theme.py          CSS + HTML helpers
views/                 login, vitals_tab, cancer_tab, admin
models/                exported notebook artifacts go here
```

## PRD coverage

| Requirement | Where |
|---|---|
| FR-1.1 login first screen · FR-1.2 no access unauthenticated | `app.py` (`st.stop()` before anything else), `views/login.py` |
| FR-1.3 hashed passwords | `core/auth.py` — PBKDF2-HMAC-SHA256, per-user salt |
| FR-1.4 session persistence · FR-1.5 logout | `st.session_state["auth"]`, Logout button |
| FR-2.1 two `st.tabs` · FR-2.2 independent state | `app.py`; each tab stores its own result key |
| FR-3.1 inputs (the notebook's vitals) · FR-3.2 validation | `views/vitals_tab.py`, `core/clinical.py` (physiological bounds) |
| FR-3.3 predict · FR-3.4 explanation | `core/vitals_model.py` — SHAP bar chart + top factors |
| FR-4.1 / 4.2 upload + validation | `core/cancer_model.py::open_validated_image` (type, size, real image, min size) |
| FR-4.3 analyze · FR-4.4 POI region | Grad-CAM → largest hot region → red POI box |
| FR-4.5 class + confidence · FR-4.6 "No suspicious finding detected" | `views/cancer_tab.py` |
| NFR models load once | `st.cache_resource` on both services |
| NFR explainability · disclaimer | every result ships context + the decision-support disclaimer |
| Prediction logs · admin user management (PRD §4, §7) | `core/auth.py`, `views/admin.py` (no patient data stored) |

## Before the defense — notebook items to double-check

1. **Tab 1 features.** The form uses exactly the vitals in the notebook's dataset: Heart Rate, Respiratory
   Rate, Body Temperature, SpO₂. The exported pipeline also expects `Unnamed: 0` and `Time (s)`; the app
   can't ask a nurse for those, so they are passed empty and the pipeline's imputer fills the training
   median. Better: drop them from `X` in the notebook and re-export (a row index isn't a vital sign and can
   inflate the ~0.998 F1). The app adapts to whichever columns the pipeline expects.
2. **Tab 2 threshold.** The decision threshold is chosen on the *test* set's PR curve. Choose it on
   the validation set (`val_flow`) and only report on test.
3. Keep the library versions in `requirements.txt` aligned with the notebook environment.
