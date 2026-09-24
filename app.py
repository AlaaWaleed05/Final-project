"""Hospital Multi-Model AI Risk Screening & Breast Cancer Detection System.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Hospital AI System", page_icon="🏥", layout="wide", initial_sidebar_state="collapsed")

from core import auth, theme  # noqa: E402  (must come after set_page_config)
from views import admin, cancer_tab, login, vitals_tab  # noqa: E402

theme.inject_css()
auth.init_db()

# FR-1.1 / FR-1.2 — nothing but the login form is reachable without a session.
if not auth.is_authenticated():
    login.render()
    st.stop()

user = auth.current_user()

with st.container(key="topbar"):
    brand, who, out = st.columns([5, 3, 1.3])
    brand.markdown('<div class="brand"><div class="mark">+</div><div class="name">Hospital AI System</div></div>', unsafe_allow_html=True)
    who.markdown(
        f'<div class="user-chip"><b>{theme.esc(user["name"])}</b><br><span>{theme.esc(auth.ROLE_LABELS.get(user["role"], user["role"]))}</span></div>',
        unsafe_allow_html=True,
    )
    out.button("Logout", key="logout", on_click=auth.logout, width="stretch")   # FR-1.5

# FR-2.1 / FR-2.2 — two independent tabs; each keeps its own state in st.session_state.
tab_vitals, tab_cancer = st.tabs(["Vital-Signs Risk Screening", "Breast Cancer Detection"])
with tab_vitals:
    vitals_tab.render()
with tab_cancer:
    cancer_tab.render()

admin.render()
theme.footer()
