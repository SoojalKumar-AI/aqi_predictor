# streamlit_app/pages/02_Explanations.py
import os, requests, pandas as pd, numpy as np, streamlit as st, altair as alt

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Explanations", page_icon="🧠", layout="wide")

# ====== THEME-AWARE STYLES ======
st.markdown("""
<style>
:root { --card-bg:#ffffff; --soft:#f7f9fc; --text:#0f1115; --muted:rgba(0,0,0,.65);
        --border:rgba(0,0,0,.08); --pill:#111; }
@media (prefers-color-scheme: dark) {
  :root { --card-bg:#17191d; --soft:#0f1115; --text:#e8eaed; --muted:rgba(255,255,255,.70);
          --border:rgba(255,255,255,.12); --pill:#e8eaed; }
}
.page-title { font-size:1.75rem; font-weight:800; letter-spacing:.2px; color:var(--text) }
.subtle { color:var(--muted) }
.card { background:var(--card-bg); border:1px solid var(--border); border-radius:16px; padding:16px 18px;
        box-shadow:0 10px 24px rgba(0,0,0,.06); color:var(--text) }
.hero { background:var(--soft); border:1px solid var(--border); border-radius:18px; padding:16px 18px;
        display:flex; align-items:center; justify-content:space-between; gap:16px; }
.pills { display:flex; gap:10px; flex-wrap:wrap }
.pill  { padding:8px 12px; border-radius:999px; border:1px solid var(--border); font-weight:700; cursor:pointer;
         text-decoration:none; color:var(--pill); }
.pill.active { background:#4f46e5; color:#fff; border-color:#4f46e5; }
.kpi { font-size:.9rem; font-weight:700; opacity:.9 }
.kpv { font-size:1.35rem; font-weight:800; margin-top:4px }
</style>
""", unsafe_allow_html=True)

# ====== PAGE HEADER ======
st.markdown("<div class='page-title'>🧠 Why the model thinks so</div>", unsafe_allow_html=True)
st.caption("Feature importance via SHAP — higher mean |SHAP| means a feature moves the prediction more on average.")

# ====== TARGETS & LABELS ======
TARGETS = [
    ("pm2_5_t+3h", "PM2.5 · next few hours"),
    ("pm2_5_t+6h", "PM2.5 · later today"),
    ("pm10_t+3h",  "PM10 · next few hours"),
    ("pm10_t+6h",  "PM10 · later today"),
]
label_map = {k:v for k,v in TARGETS}

# ====== HELPERS ======
@st.cache_data(show_spinner=False, ttl=600)
def fetch_shap(target: str, k=40) -> pd.DataFrame | None:
    try:
        r = requests.get(f"{API}/explanations", params={"target": target, "top_k": k}, timeout=30)
        r.raise_for_status()
        data = r.json().get("features", [])
        df = pd.DataFrame(data) if data else None
        # normalize possible column names
        if df is not None:
            if "feature" not in df.columns:
                df.rename(columns={df.columns[0]: "feature"}, inplace=True)
            if "mean_abs_shap" not in df.columns:
                # guess the shap value column
                for c in df.columns:
                    if c.lower().startswith("mean") or c.lower().endswith("shap"):
                        df.rename(columns={c: "mean_abs_shap"}, inplace=True)
                        break
        return df
    except Exception:
        return None

def feature_family(name: str) -> str:
    n = name.lower()
    if "_lag_" in n: return "Lagged"
    if "mean_6h" in n or "std_6h" in n or "min_6h" in n or "max_6h" in n: return "Rolling 6h"
    if "mean_12h" in n or "std_12h" in n or "min_12h" in n or "max_12h" in n: return "Rolling 12h"
    if "avg_3h" in n: return "Short avg (3h)"
    if n in {"hour","weekday","month","is_weekend","hour_sin","hour_cos","wday_sin","wday_cos","month_sin","month_cos"}:
        return "Calendar/Cycle"
    if "wind_dir_" in n: return "Wind direction"
    if any(x in n for x in ["temperature_2m","relative_humidity_2m","wind_speed_10m"]): return "Weather"
    if any(x in n for x in ["pm2_5","pm10","ozone","nitrogen_dioxide","carbon_monoxide","sulphur_dioxide"]): return "Pollutants/Derived"
    return "Other"

def cum_coverage(values: np.ndarray, k: int) -> float:
    v = np.array(values, dtype=float)
    if len(v) == 0: return 0.0
    v = np.abs(v)
    v_sorted = np.sort(v)[::-1]
    k = min(k, len(v_sorted))
    return float(v_sorted[:k].sum() / (v_sorted.sum() + 1e-12))

def shap_chart(df: pd.DataFrame, title: str, top_k: int, search: str | None):
    if df is None or df.empty:
        st.info("No explanation data available.")
        return

    # filter
    work = df.copy()
    if search:
        s = search.lower().strip()
        work = work[work["feature"].str.lower().str.contains(s)]

    # top-k
    work = work.sort_values("mean_abs_shap", ascending=False).head(top_k)
    work["family"] = work["feature"].apply(feature_family)

    # bar + cumulative line
    bar = (
        alt.Chart(work)
        .transform_window(rank="rank()", sort=[alt.SortField("mean_abs_shap", order="descending")])
        .mark_bar()
        .encode(
            x=alt.X("mean_abs_shap:Q", title="Mean |SHAP| (impact on prediction)"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            color=alt.Color("family:N", title="Group",
                            scale=alt.Scale(scheme="tableau20")),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("family:N",  title="Group"),
                alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".5f"),
            ],
        )
        .properties(height=420, title=title)
    )

    # cumulative %
    cum = work.copy()
    cum["rank"] = np.arange(1, len(cum) + 1)
    cum["cum_share"] = cum["mean_abs_shap"].abs().cumsum() / cum["mean_abs_shap"].abs().sum()
    line = (
        alt.Chart(cum)
        .mark_line(point=True)
        .encode(
            x=alt.X("rank:Q", title="Top-k features"),
            y=alt.Y("cum_share:Q", title="Cumulative importance", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("rank:Q", title="k"), alt.Tooltip("cum_share:Q", title="Coverage", format=".0%")],
        )
        .properties(height=200, title="How much of the explanation is covered by the top-k?")
    )

    st.altair_chart(bar, use_container_width=True)
    st.altair_chart(line, use_container_width=True)
# ====== QUICK TARGET PICKER (no navigation) ======
query_param_target = st.query_params.get("target", None)
default_idx = 0
keys = [t for t, _ in TARGETS]
if query_param_target in keys:
    default_idx = keys.index(query_param_target)

# nice header row
cols = st.columns([3, 1.6])
with cols[0]:
    st.markdown(
        "<div class='hero'><div><div class='page-title' style='font-size:1.35rem;'>Model explanations (global)</div>"
        "<div class='subtle'>These are training-time SHAP summaries for each target. Larger bars = bigger average influence.</div></div></div>",
        unsafe_allow_html=True
    )

st.write("")  # spacer

# radio avoids page navigation; horizontal for “pill” vibe
picked = st.radio(
    "Choose a forecast",
    options=keys,
    index=default_idx,
    format_func=lambda k: label_map[k],
    horizontal=True,
)

# keep URL in sync without navigating away
st.query_params.update({"target": picked})

# ====== CONTROLS ======
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.4,1,1,1.6])
with ctrl1:
    top_k = st.slider("Top features", 5, 40, 20, step=1)
with ctrl2:
    show_table = st.toggle("Show table", value=False)
with ctrl3:
    group_tip = st.toggle("Show grouping tip", value=False)
with ctrl4:
    search = st.text_input("Filter features (search)", "")

if group_tip:
    st.info("Grouping is inferred from names: e.g., *_lag_* → Lagged, mean_6h/std_6h → Rolling 6h, calendar vars "
            "(hour, weekday, month, *_sin/cos) → Calendar/Cycle, weather vars → Weather, etc.")

# ====== FETCH & SUMMARIZE ======
df = fetch_shap(picked, k=40)
if df is None or df.empty:
    st.warning("No SHAP data returned. Try retraining with SHAP enabled.")
else:
    # KPIs
    coverage = cum_coverage(df["mean_abs_shap"].values, top_k)
    top_feat = df.sort_values("mean_abs_shap", ascending=False).iloc[0]["feature"]
    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
    with kpi_c1:
        st.markdown("<div class='card'><div class='kpi'>Explained by top-k</div>"
                    f"<div class='kpv'>{coverage*100:.0f}%</div></div>", unsafe_allow_html=True)
    with kpi_c2:
        st.markdown("<div class='card'><div class='kpi'>Top driver</div>"
                    f"<div class='kpv'>{top_feat}</div></div>", unsafe_allow_html=True)
    with kpi_c3:
        st.markdown("<div class='card'><div class='kpi'>Features available</div>"
                    f"<div class='kpv'>{len(df):,}</div></div>", unsafe_allow_html=True)

    st.write("")  # spacer

    shap_chart(df, f"{label_map[picked]} — top {top_k}", top_k=top_k, search=search)

    if show_table:
        st.markdown("#### Details")
        st.dataframe(
            df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{picked}_shap_top{max(top_k, len(df))}.csv",
            mime="text/csv"
        )

# ====== FOOTNOTE ======
st.caption("Tip: High mean |SHAP| doesn’t mean causation. It means the model used that feature a lot to move predictions.")
