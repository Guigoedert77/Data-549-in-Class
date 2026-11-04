"""Liquid-glass styling for Cabin Voice (CSS injected into Streamlit)."""

FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', 'Helvetica Neue', sans-serif"

COLORS = {'negative': '#FF6B5E', 'neutral': '#C9CED8', 'positive': '#5BD6A0', 'accent': '#F26B2A'}


def css():
    faces = ''.join(
        f"@font-face{{font-family:'Inter';font-weight:{w};font-display:swap;"
        f"src:url('app/static/fonts/inter-latin-{w}-normal.woff2') format('woff2');}}"
        for w in (400, 500, 600, 700))
    return faces + f"""
:root {{
  --ink:#FFFFFF; --muted:rgba(255,255,255,.80); --faint:rgba(255,255,255,.62);
  --rim:rgba(255,255,255,.40); --rim-soft:rgba(255,255,255,.30);
  --accent:{COLORS['accent']}; --neg:{COLORS['negative']}; --neu:{COLORS['neutral']}; --pos:{COLORS['positive']};
  --font:{FONT};
}}

/* ---------- page + photo background ---------- */
.stApp {{
  background: #15110e url('app/static/bg.jpg') 71% 30% / auto 117% no-repeat fixed !important;
  font-family:var(--font);
}}
.stApp::before {{ content:''; position:fixed; inset:0; pointer-events:none;
  background:radial-gradient(90% 80% at 50% 50%, rgba(0,0,0,0) 0%, rgba(0,0,0,.28) 100%); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:transparent !important; }}
[data-testid="stHeader"] {{ background:transparent !important; height:0 !important; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {{ display:none !important; }}
html, body, p, label, div, input, textarea, button, li {{ font-family:var(--font) !important; -webkit-font-smoothing:antialiased; }}
[data-testid="stIconMaterial"] {{ font-family:'Material Symbols Rounded' !important; }}
p, label, span, li {{ color:var(--ink); }}

/* ---------- the central glass sheet ---------- */
[data-testid="stMainBlockContainer"] {{
  max-width:1180px !important; margin:110px auto 60px auto !important; padding:2.3rem 2.4rem 2.4rem !important;
  border-radius:38px; position:relative;
  background:rgba(48,38,32,.22);
  backdrop-filter:blur(22px) brightness(1.08) saturate(125%); -webkit-backdrop-filter:blur(22px) brightness(1.08) saturate(125%);
  box-shadow: inset 0 0 0 1.5px var(--rim), inset 0 1px 0 rgba(255,255,255,.6), 0 40px 90px -40px rgba(0,0,0,.65);
}}

/* ---------- inner glass panels (keyed containers) ---------- */
.st-key-stats, [class*="st-key-panel"] {{
  background:rgba(40,32,28,.16) !important; border:none !important; border-radius:26px !important; padding:1.4rem 1.5rem !important;
  backdrop-filter:blur(14px) brightness(1.06); -webkit-backdrop-filter:blur(14px) brightness(1.06);
  box-shadow: inset 0 0 0 1.5px var(--rim-soft), inset 0 1px 0 rgba(255,255,255,.5);
}}
.st-key-stats {{ padding:1.1rem 1.5rem !important; }}

/* ---------- typography ---------- */
.cv-title {{ font-size:3rem; font-weight:600; letter-spacing:-.03em; line-height:1; margin:0; color:#fff; }}
.cv-sub {{ color:var(--muted); font-size:1rem; margin:.55rem 0 0 0; }}
.cv-h {{ font-size:1.45rem; font-weight:600; letter-spacing:-.02em; margin:0 0 .4rem 0; color:#fff; }}
.cv-small {{ color:var(--faint); font-size:.84rem; line-height:1.5; }}

/* ---------- stats strip ---------- */
.cv-stats {{ display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
.cv-stat {{ display:flex; align-items:center; gap:.9rem; }}
.cv-ring {{ width:54px; height:54px; border-radius:50%; display:grid; place-items:center; box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.8); }}
.cv-ring svg {{ width:22px; height:22px; stroke:#fff; fill:none; stroke-width:1.7; stroke-linecap:round; stroke-linejoin:round; }}
.cv-stat small {{ display:block; color:var(--muted); font-size:.9rem; }}
.cv-stat b {{ font-size:1.3rem; font-weight:600; color:#fff; }}

/* ---------- segmented control (view switch) ---------- */
.st-key-view [data-testid="stButtonGroup"] {{ background:rgba(255,255,255,.08); border-radius:999px; padding:5px; gap:4px;
  box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.36); display:inline-flex; }}
.st-key-view button {{ border:none !important; border-radius:999px !important; background:transparent !important; color:var(--muted) !important;
  padding:.45rem 1.1rem !important; min-height:0 !important; }}
.st-key-view button[kind="segmented_controlActive"], .st-key-view button[aria-checked="true"], .st-key-view button[data-testid="stBaseButton-segmented_controlActive"] {{
  background:rgba(255,255,255,.22) !important; color:#fff !important; box-shadow:inset 0 1px 0 rgba(255,255,255,.4); }}
.st-key-view button p {{ font-size:.92rem !important; }}

/* ---------- floating nav pill ---------- */
.st-key-navpill {{ position:absolute !important; left:-118px; top:300px; z-index:50; width:84px !important;
  padding:16px 0 !important; border-radius:42px; display:flex; flex-direction:column; align-items:center; gap:14px;
  background:rgba(255,255,255,.09); backdrop-filter:blur(18px) brightness(1.08); -webkit-backdrop-filter:blur(18px) brightness(1.08);
  box-shadow:inset 0 0 0 1.5px var(--rim), inset 0 1px 0 rgba(255,255,255,.6); }}
.st-key-navpill [data-testid="stElementContainer"] {{ width:auto !important; }}
.st-key-navpill button {{ width:56px !important; height:56px !important; min-height:0 !important; border-radius:50% !important; padding:0 !important;
  border:none !important; background:transparent !important; color:rgba(255,255,255,.88) !important; }}
.st-key-navpill button:hover {{ background:rgba(255,255,255,.12) !important; }}
.st-key-navpill button[kind="primary"] {{ background:var(--accent) !important; color:#fff !important;
  box-shadow:0 10px 24px -8px rgba(242,107,42,.7), inset 0 1px 0 rgba(255,255,255,.45) !important; }}
.st-key-navpill button span[data-testid="stIconMaterial"] {{ font-size:26px !important; }}
@media (max-width: 1420px) {{ .st-key-navpill {{ display:none !important; }} }}

/* ---------- inputs ---------- */
 [data-testid="stSelectbox"] [data-baseweb="select"] * {{ background:transparent !important; color:#fff !important; }}
.stTextArea textarea, [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
  background:rgba(255,255,255,.08) !important; color:#fff !important; border:none !important; border-radius:16px !important;
  box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.3), inset 0 1px 4px rgba(0,0,0,.25) !important; font-size:1.02rem !important; line-height:1.5 !important; }}
.stTextArea textarea:focus {{ box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.7), 0 0 0 3px rgba(242,107,42,.35) !important; }}
[data-testid="stTextArea"] > div, [data-baseweb="textarea"], [data-baseweb="base-input"] {{ background:transparent !important; border:none !important; }}
[data-testid="stWidgetLabel"] p {{ color:var(--muted) !important; font-size:.88rem !important; }}
[data-baseweb="select"] svg {{ fill:#fff !important; }}
[data-baseweb="popover"] ul {{ background:#2a2320 !important; }}

/* example pills */
.st-key-examples [data-testid="stButtonGroup"] {{ gap:8px; flex-wrap:wrap; }}
.st-key-examples button {{ border-radius:999px !important; background:rgba(255,255,255,.07) !important; border:none !important;
  box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.32) !important; color:#fff !important; padding:.3rem .9rem !important; min-height:0 !important; }}
.st-key-examples button:hover {{ background:rgba(255,255,255,.16) !important; }}

/* primary CTA */
[data-testid="stBaseButton-primary"] {{ background:var(--accent) !important; border:none !important; border-radius:999px !important;
  box-shadow:0 14px 30px -12px rgba(242,107,42,.75), inset 0 1px 0 rgba(255,255,255,.45) !important; padding:.75rem !important; }}
[data-testid="stBaseButton-primary"] p {{ color:#fff !important; font-weight:600 !important; font-size:1.05rem !important; }}
[data-testid="stBaseButton-primary"]:hover {{ filter:brightness(1.07); }}

/* ---------- verdict ---------- */
.cv-verdict {{ display:grid; grid-template-columns:1fr auto; align-items:center; gap:1rem; }}
.cv-vlabel {{ color:var(--muted); font-size:.95rem; }}
.cv-vword {{ font-size:3.6rem; font-weight:600; letter-spacing:-.04em; line-height:1.05; }}
.cv-probs {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:1.2rem; }}
.cv-prob {{ border-radius:20px; padding:1rem 1.1rem 1.1rem; background:rgba(255,255,255,.07); box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.32); }}
.cv-prob .p {{ font-size:2.1rem; font-weight:600; letter-spacing:-.03em; color:#fff; line-height:1.1; }}
.cv-bar {{ height:8px; border-radius:99px; background:rgba(255,255,255,.14); margin:.7rem 0 .55rem; overflow:hidden; }}
.cv-bar i {{ display:block; height:100%; border-radius:99px; }}
.cv-prob span {{ color:#fff; font-size:.95rem; }}

/* ---------- why ---------- */
.cv-why p {{ color:rgba(255,255,255,.88); font-size:.98rem; line-height:1.55; margin:0 0 .9rem 0; }}
.cv-row {{ display:grid; grid-template-columns:minmax(120px, 34%) 1fr; align-items:center; gap:14px; margin:.5rem 0; font-size:.93rem; }}
.cv-row .t {{ color:rgba(255,255,255,.88); text-align:right; }}
.cv-track {{ position:relative; height:12px; }}
.cv-track .mid {{ position:absolute; left:35%; top:-5px; bottom:-5px; width:1px; background:rgba(255,255,255,.35); }}
.cv-track i {{ position:absolute; top:0; height:12px; border-radius:99px; }}
.cv-legend {{ display:flex; justify-content:space-between; color:var(--faint); font-size:.78rem; margin-top:.5rem; padding-left:34%; }}

/* ---------- benchmark table ---------- */
.cv-table {{ width:100%; border-collapse:collapse; font-size:.95rem; }}
.cv-table th {{ text-align:left; color:var(--muted); font-weight:500; padding:.5rem .6rem; border-bottom:1px solid rgba(255,255,255,.2); }}
.cv-table td {{ padding:.6rem; border-bottom:1px solid rgba(255,255,255,.1); color:#fff; }}
[data-testid="stMarkdownContainer"] .cv-table th, [data-testid="stMarkdownContainer"] .cv-table td {{ border-left:none !important; border-right:none !important; border-top:none !important; }}
.cv-table td.num {{ font-variant-numeric:tabular-nums; }}
.cv-pill {{ display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.82rem; box-shadow:inset 0 0 0 1px rgba(255,255,255,.3); }}

/* compare + topics */
.cv-kpis {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-top:.6rem; }}
.cv-kpi {{ border-radius:18px; padding:.8rem 1rem; background:rgba(255,255,255,.07); box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.28); }}
.cv-kpi small {{ display:block; color:var(--muted); font-size:.8rem; }}
.cv-kpi b {{ font-size:1.5rem; font-weight:600; color:#fff; letter-spacing:-.02em; }}
.cv-airline {{ font-size:1.9rem; font-weight:600; letter-spacing:-.03em; color:#fff; }}
.cv-chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:.4rem; }}
.cv-chip {{ padding:.25rem .7rem; border-radius:999px; font-size:.85rem; color:#fff; background:rgba(255,255,255,.07); box-shadow:inset 0 0 0 1px rgba(255,255,255,.3); }}
.cv-chip.hot {{ background:rgba(242,107,42,.28); box-shadow:inset 0 0 0 1px rgba(242,107,42,.7); }}
.cv-quote {{ border-left:2px solid var(--accent); padding:.2rem 0 .2rem .8rem; margin:.5rem 0; color:rgba(255,255,255,.88); font-size:.92rem; line-height:1.45; }}
[data-testid="stFileUploaderDropzone"] {{ background:rgba(255,255,255,.07) !important; border:1.5px dashed rgba(255,255,255,.4) !important; border-radius:18px !important; }}
[data-testid="stDownloadButton"] button, .st-key-panel_batch [data-testid="stBaseButton-secondary"] {{ border-radius:999px !important; background:rgba(255,255,255,.1) !important;
  border:none !important; box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.4) !important; color:#fff !important; }}
[data-testid="stDataFrame"] {{ border-radius:16px; overflow:hidden; }}
[data-testid="stImage"] img {{ border-radius:16px; }}

/* misc */
[data-testid="stCaptionContainer"] p {{ color:var(--faint) !important; }}
[data-testid="stAlert"] {{ background:rgba(255,255,255,.08) !important; border-radius:16px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
@media (max-width: 760px) {{
  [data-testid="stMainBlockContainer"] {{ margin:16px auto !important; padding:1.4rem 1.1rem !important; border-radius:26px; }}
  .cv-title {{ font-size:2.2rem; }} .cv-vword {{ font-size:2.6rem; }} .cv-probs {{ grid-template-columns:1fr; }}
}}
"""


ICONS = {
    'chat': '<svg viewBox="0 0 24 24"><path d="M4 5h16v11H8l-4 4z"/><path d="M8 9h8M8 12h5"/></svg>',
    'check': '<svg viewBox="0 0 24 24"><path d="M5 12l4 4 10-10"/></svg>',
    'plane': '<svg viewBox="0 0 24 24"><path d="M2 16l20-8-6 12-3-5z"/></svg>',
}
