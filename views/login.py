"""Login screen (FR-1.1) — the only thing an unauthenticated session can reach."""
import streamlit as st

from core import auth


def render() -> None:
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        with st.container(key="login_card"):
            st.markdown(
                '<div class="login-logo">+</div>'
                '<div class="login-title">Hospital AI System</div>'
                '<div class="login-sub">Sign in to continue</div>',
                unsafe_allow_html=True,
            )
            with st.form("login_form", border=False):
                username = st.text_input("Username", placeholder="e.g. staff_sara", autocomplete="username")
                password = st.text_input("Password", type="password", placeholder="••••••••••", autocomplete="current-password")
                submitted = st.form_submit_button("LOG IN", type="primary", width="stretch")
            if submitted:
                ok, msg = auth.login(username, password)
                if ok:
                    st.rerun()
                st.error(msg)
            st.markdown('<div class="login-foot">Authorized hospital staff only</div>', unsafe_allow_html=True)
