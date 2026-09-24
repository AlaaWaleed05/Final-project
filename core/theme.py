"""Look & feel — one place for CSS and HTML snippets so the views stay readable."""
from __future__ import annotations

import html

import streamlit as st

from .config import AQUA, BLUE, CORAL, DISCLAIMER, GREEN, INK, SUN

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

:root {{
  --blue: {BLUE}; --aqua: {AQUA}; --coral: {CORAL}; --sun: {SUN}; --green: {GREEN}; --ink: {INK};
  --line: #D9E3FF; --mist: #EEF3FF;
}}
html, body, [class*="st-"], .stApp {{ font-family: 'Figtree', 'Segoe UI', system-ui, sans-serif; }}
h1, h2, h3, .display {{ font-family: 'Sora', 'Figtree', sans-serif !important; color: var(--ink); letter-spacing: -0.01em; }}

/* page chrome */
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.stApp {{
  background:
    radial-gradient(900px 420px at 8% -8%, rgba(47,91,255,.16), transparent 60%),
    radial-gradient(760px 380px at 100% 0%, rgba(0,184,169,.16), transparent 60%),
    radial-gradient(700px 500px at 60% 110%, rgba(255,176,32,.14), transparent 60%),
    #F3F6FF;
}}
.block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1180px; }}

/* top bar */
.st-key-topbar {{
  background: linear-gradient(100deg, var(--blue) 0%, #5B4BFF 55%, #00A7C9 100%);
  border-radius: 20px; padding: .7rem 1.2rem; margin-bottom: 1.1rem;
  box-shadow: 0 10px 28px rgba(47,91,255,.28);
}}
.st-key-topbar [data-testid="stHorizontalBlock"] {{ align-items: center; }}
.brand {{ display: flex; align-items: center; gap: .7rem; color: #fff; }}
.brand .mark {{ width: 38px; height: 38px; border-radius: 12px; background: #fff; color: var(--blue);
  display:flex; align-items:center; justify-content:center; font: 800 1.35rem 'Sora', sans-serif; }}
.brand .name {{ font: 700 1.2rem 'Sora', sans-serif; }}
.user-chip {{ color: #fff; text-align: right; line-height: 1.15; }}
.user-chip b {{ font-weight: 700; }}
.user-chip span {{ font-size: .8rem; opacity: .85; }}
.st-key-topbar [data-testid="stBaseButton-secondary"] {{
  background: rgba(255,255,255,.18); color: #fff; border: 1px solid rgba(255,255,255,.55); border-radius: 12px; font-weight: 700;
}}
.st-key-topbar [data-testid="stBaseButton-secondary"]:hover {{ background: #fff; color: var(--blue); border-color: #fff; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .6rem; border-bottom: none; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs button[data-baseweb="tab"] {{
  background: #fff; border: 1.5px solid var(--line); border-radius: 14px; padding: .55rem 1.15rem; height: auto;
  font: 700 .98rem 'Figtree', sans-serif; color: var(--ink);
}}
.stTabs button[data-baseweb="tab"]:hover {{ border-color: var(--blue); }}
.stTabs button[aria-selected="true"] {{
  background: linear-gradient(100deg, var(--blue), #5B4BFF); color: #fff; border-color: transparent;
  box-shadow: 0 6px 16px rgba(47,91,255,.32);
}}
.stTabs button[aria-selected="true"] p {{ color: #fff; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.1rem; }}

/* form + cards */
[data-testid="stForm"], .st-key-upload_card {{
  background: #fff; border: 1.5px solid var(--line); border-radius: 22px; padding: 1.3rem 1.4rem;
  box-shadow: 0 8px 26px rgba(20,32,75,.06);
}}
.section-title {{ font: 700 1.25rem 'Sora', sans-serif; color: var(--ink); margin: .1rem 0 .7rem; }}
.hint {{ color: #5C6A96; font-size: .86rem; }}

/* buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
  background: linear-gradient(100deg, var(--blue), #5B4BFF); border: none; color: #fff; font-weight: 800;
  letter-spacing: .04em; border-radius: 14px; padding: .6rem 1.6rem; box-shadow: 0 8px 18px rgba(47,91,255,.32);
  transition: transform .12s ease, box-shadow .12s ease;
}}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{
  transform: translateY(-1px); box-shadow: 0 12px 22px rgba(47,91,255,.38); color: #fff;
}}
[data-testid="stBaseButton-primary"]:focus-visible, [data-testid="stBaseButton-primaryFormSubmit"]:focus-visible {{ outline: 3px solid var(--sun); outline-offset: 2px; }}
[data-testid="stBaseButton-secondary"] {{ border-radius: 12px; font-weight: 700; border: 1.5px solid var(--line); }}

/* inputs */
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-baseweb="select"] > div {{ border-radius: 12px; }}
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed var(--blue); border-radius: 16px; background: var(--mist); }}

/* result cards */
.res {{ border-radius: 20px; padding: 1.1rem 1.3rem; margin-top: 1rem; border: 1.5px solid; }}
.res .lab {{ font: 700 .9rem 'Figtree', sans-serif; margin-bottom: .1rem; }}
.res .big {{ font: 800 2.3rem 'Sora', sans-serif; line-height: 1.1; margin: .1rem 0 .3rem; }}
.res .meta {{ font-size: .9rem; color: #43507D; }}
.res.low   {{ background: #E9FBF3; border-color: #9BE3C2; }} .res.low .lab, .res.low .big {{ color: #0B8F55; }}
.res.high  {{ background: #FFF0F3; border-color: #FFB3C1; }} .res.high .lab, .res.high .big {{ color: #E0234A; }}
.res.clear {{ background: #E9FBF3; border-color: #9BE3C2; }} .res.clear .lab, .res.clear .big {{ color: #0B8F55; }}
.res.flag  {{ background: #FFF0F3; border-color: #FFB3C1; }} .res.flag .lab, .res.flag .big {{ color: #E0234A; }}
.meter {{ height: 12px; border-radius: 99px; background: #E4E9FA; overflow: hidden; margin: .5rem 0 .2rem; }}
.meter > i {{ display: block; height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--aqua), var(--sun), var(--coral)); }}

/* reference chips */
.chips {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .3rem 0 .2rem; }}
.chip {{ border-radius: 999px; padding: .3rem .8rem; font-size: .82rem; font-weight: 600; border: 1.5px solid; background: #fff; }}
.chip.normal {{ border-color: #9BE3C2; color: #0B8F55; }}
.chip.high   {{ border-color: #FFB3C1; color: #E0234A; }}
.chip.low    {{ border-color: #FFD48A; color: #B26A00; }}

/* image panel */
.placeholder {{
  border: 2px dashed #B9C8FF; border-radius: 22px; min-height: 420px; display: flex; align-items: center; justify-content: center;
  text-align: center; color: #5C6A96; background: rgba(255,255,255,.7); padding: 1.5rem;
}}
[data-testid="stImage"] img {{ border-radius: 18px; box-shadow: 0 10px 28px rgba(20,32,75,.16); }}

/* login */
.st-key-login_card {{
  background: #fff; border-radius: 26px; padding: 2rem 2rem 1.4rem; border: 1.5px solid var(--line);
  box-shadow: 0 24px 60px rgba(47,91,255,.20); margin-top: 6vh;
}}
.st-key-login_card [data-testid="stForm"] {{ border: none; box-shadow: none; padding: 0; background: transparent; }}
.login-logo {{ width: 60px; height: 60px; border-radius: 50%; margin: 0 auto .7rem; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, var(--blue), var(--aqua)); color:#fff; font: 800 2rem 'Sora', sans-serif; box-shadow: 0 10px 22px rgba(47,91,255,.35); }}
.login-title {{ text-align:center; font: 800 1.45rem 'Sora', sans-serif; color: var(--ink); }}
.login-sub {{ text-align:center; color:#5C6A96; margin-bottom: 1rem; }}
.login-foot {{ text-align:center; color:#7A87B3; font-size:.8rem; margin-top:.8rem; }}

.footer-note {{ text-align:center; color:#5C6A96; font-size:.82rem; margin-top: 2rem; }}

@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
@media (max-width: 720px) {{ .res .big {{ font-size: 1.8rem; }} .block-container {{ padding-left: .8rem; padding-right: .8rem; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def esc(text: object) -> str:
    return html.escape(str(text))


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{esc(text)}</div>', unsafe_allow_html=True)


def result_card(kind: str, label: str, big: str, meta_html: str = "") -> None:
    """kind: low | high | clear | flag."""
    st.markdown(
        f'<div class="res {kind}"><div class="lab">{esc(label)}</div><div class="big">{esc(big)}</div>'
        f'<div class="meta">{meta_html}</div></div>',
        unsafe_allow_html=True,
    )


def risk_meter(probability: float) -> None:
    pct = max(0.0, min(1.0, probability)) * 100
    st.markdown(
        f'<div class="hint">Probability of high risk: <b>{pct:.1f}%</b></div>'
        f'<div class="meter"><i style="width:{pct:.1f}%"></i></div>',
        unsafe_allow_html=True,
    )


def chips(rows: list[dict]) -> None:
    body = "".join(
        f'<span class="chip {r["status"]}" title="Adult reference {esc(r["range"])}">{esc(r["label"])}: {esc(r["value"])} · {r["status"]}</span>'
        for r in rows
    )
    st.markdown(f'<div class="chips">{body}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<div class="hint" style="margin-top:.6rem">⚕️ {esc(DISCLAIMER)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer-note">Hospital AI System · Graduation Project (AI / Data Science)<br>{esc(DISCLAIMER)}</div>', unsafe_allow_html=True)
