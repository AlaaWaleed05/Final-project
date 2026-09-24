"""System Admin tools (PRD §4: manage user accounts, monitor system usage). Visible to role 'admin' only."""
import pandas as pd
import streamlit as st

from core import auth


def render() -> None:
    if (auth.current_user() or {}).get("role") != "admin":
        return
    with st.expander("🛠️ Admin — user accounts & usage"):
        view = st.radio("Section", ["Usage log", "Users", "Add user"], horizontal=True, label_visibility="collapsed", key="admin_view")

        if view == "Usage log":
            logs = auth.recent_logs(100)
            if logs:
                st.dataframe(pd.DataFrame(logs), width="stretch", hide_index=True)
                st.caption("The log keeps who ran which tab and the outcome — never patient data or images.")
            else:
                st.info("No predictions yet.")

        elif view == "Users":
            users = pd.DataFrame(auth.list_users())
            users["active"] = users["active"].astype(bool)
            st.dataframe(users, width="stretch", hide_index=True)
            me = auth.current_user()["username"]
            others = [u["username"] for u in auth.list_users() if u["username"] != me]
            if others:
                c1, c2 = st.columns([2, 1])
                target = c1.selectbox("Account", others, key="admin_target")
                c2.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
                current = next(u for u in auth.list_users() if u["username"] == target)["active"]
                if c2.button("Deactivate" if current else "Reactivate", key="admin_toggle"):
                    auth.set_user_active(target, not current)
                    st.rerun()

        else:
            with st.form("add_user_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                username = c1.text_input("Username", placeholder="e.g. staff_nour")
                full_name = c2.text_input("Full name")
                role = c1.selectbox("Role", list(auth.ROLE_LABELS), format_func=lambda r: auth.ROLE_LABELS[r])
                password = c2.text_input("Password (min 8 characters)", type="password")
                if st.form_submit_button("Create user", type="primary"):
                    ok, msg = auth.create_user(username, password, role, full_name)
                    (st.success if ok else st.error)(msg)
