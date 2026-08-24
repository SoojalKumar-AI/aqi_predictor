# streamlit_app/pages/03_EDA.py
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="EDA", page_icon="🔎", layout="wide")

# ====================== STYLES ======================
st.markdown("""
<style>
:root { --card-bg:#ffffff; --soft:#f7f9fc; --text:#0f1115; --muted:rgba(0,0,0,.65); --border:rgba(0,0,0,.08); }
@media (prefers-color-scheme: dark) {
  :root { --card-bg:#17191d; --soft:#0f1115; --text:#e8eaed; --muted:rgba(255,255,255,.70); --border:rgba(255,255,255,.12); }
}
.page-title { font-size:1.6rem; font-weight:800; color:var(--text) }
.subtle { color:var(--muted) }
.card { background:var(--card-bg); border:1px solid var(--border); border-radius:16px; padding:14px 16px; }
.k { font-weight:700; font-size:.9rem; opacity:.9 }
.v { font-size:1.25rem; font-weight:800; margin-top:2px }
.tip { font-size:.92rem; color:var(--muted); }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='page-title'>🔎 Explore the data</div>", unsafe_allow_html=True)
st.caption("Two views: **Raw time-series** for recent behavior and **Training table** for model inputs. Clean, minimal, and useful.")

# ====================== HELPERS ======================
def pm_color_scale():
    return alt.Scale(domain=["pm2_5", "pm10"], range=["#2E86C1", "#E67E22"])

@st.cache_data(ttl=300, show_spinner=True)
def load_raw(hours: int):
    try:
        r = requests.get(f"{API}/eda/raw", params={"hours": hours}, timeout=60)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        return pd.DataFrame(), {}, f"HTTP error: {e}"
    cols = js.get("columns", [])
    recs = js.get("records", [])
    meta = js.get("meta", {})
    if not recs:
        return pd.DataFrame(), meta, "No rows returned."
    df = pd.DataFrame.from_records(recs, columns=cols)
    if "time" not in df.columns:
        return pd.DataFrame(), meta, "Missing 'time' column in /eda/raw payload."
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    return df, meta, None

@st.cache_data(ttl=300, show_spinner=True)
def load_training(target: str, limit: int):
    try:
        r = requests.get(f"{API}/eda/training", params={"target": target, "limit": limit}, timeout=60)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        return pd.DataFrame(), {}, f"HTTP error: {e}"
    cols = js.get("columns", [])
    recs = js.get("records", [])
    meta = js.get("meta", {})
    if not recs:
        return pd.DataFrame(), meta, "No rows returned."
    df = pd.DataFrame.from_records(recs, columns=cols)
    # Do NOT assume 'time' exists here. Training CSVs often don't include it.
    # Ensure target column exists
    tgt = meta.get("target")
    if tgt and tgt not in df.columns:
        return df, meta, f"Training dataset missing expected target column '{tgt}'."
    return df, meta, None

def density_chart(df: pd.DataFrame, col: str, title: str, fmt=".1f"):
    d = df[[col]].dropna()
    if d.empty:
        return None
    ch = (
        alt.Chart(d)
        .transform_density(col, as_=[col, "density"])
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X(f"{col}:Q", title=title),
            y=alt.Y("density:Q", title="Density"),
            tooltip=[alt.Tooltip(f"{col}:Q", format=fmt)]
        )
        .properties(height=160)
    )
    return ch

def corr_heatmap(df: pd.DataFrame, cols: list[str], title: str):
    num = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(num) < 2:
        return None
    corr = df[num].corr().stack().reset_index()
    corr.columns = ["x", "y", "value"]
    ch = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("x:N", title=None),
            y=alt.Y("y:N", title=None),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Pearson r"),
            tooltip=["x:N", "y:N", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=280, title=title)
    )
    return ch

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("View")
    view = st.radio("Data source", ["Raw time-series", "Training table"], index=0)

    if view == "Raw time-series":
        hours = st.slider("Window (hours)", 24, 24*30, 168, step=24)
        show_debug_raw = st.toggle("Show API debug", value=False)
    else:
        target = st.selectbox(
            "Target dataset",
            ["pm2_5_t+3h", "pm2_5_t+6h", "pm10_t+3h", "pm10_t+6h"],
            index=0
        )
        limit = st.slider("Rows (tail)", 1000, 200000, 20000, step=1000)
        show_debug_train = st.toggle("Show API debug", value=False)

# ====================== RAW VIEW ======================
if view == "Raw time-series":
    df, meta, err = load_raw(hours)
    if show_debug_raw:
        with st.expander("API payload (raw)"):
            st.write({"meta": meta, "shape": None if df is None else df.shape, "columns": list(df.columns) if not df.empty else []})
    if err:
        st.error(err)
        st.stop()
    if df.empty:
        st.info("No data to display.")
        st.stop()

    # quick validation
    pm_cols = [c for c in ["pm2_5", "pm10"] if c in df.columns]
    if not pm_cols:
        st.error("Need at least one PM column (pm2_5 or pm10) in /eda/raw.")
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='card'><div class='k'>Rows in window</div><div class='v'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='card'><div class='k'>Coverage</div><div class='v'>{meta.get('start','?')} → {meta.get('end','?')}</div></div>", unsafe_allow_html=True)
    with k3:
        v = df["pm2_5"].mean() if "pm2_5" in df.columns else np.nan
        st.markdown("<div class='card'><div class='k'>Mean PM2.5 (µg/m³)</div>"
                    f"<div class='v'>{('—' if pd.isna(v) else f'{v:.1f}')}</div></div>", unsafe_allow_html=True)
    with k4:
        v = df["pm10"].mean() if "pm10" in df.columns else np.nan
        st.markdown("<div class='card'><div class='k'>Mean PM10 (µg/m³)</div>"
                    f"<div class='v'>{('—' if pd.isna(v) else f'{v:.1f}')}</div></div>", unsafe_allow_html=True)

    st.divider()

    # 1) Trend with 6h smoothing
    st.subheader("📈 Trend with 6h smoothing")
    st.markdown("<div class='tip'>See short-term movement and noise-reduced trend. Use this to spot spikes and their duration.</div>", unsafe_allow_html=True)
    trend = df[["time"] + pm_cols].set_index("time").sort_index()
    smooth = trend.rolling("6h", min_periods=1).mean()
    # base line
    c_base = (
        alt.Chart(trend.reset_index().melt("time", var_name="Series", value_name="value"))
        .mark_line()
        .encode(
            x="time:T",
            y=alt.Y("value:Q", title="µg/m³"),
            color=alt.Color("Series:N", scale=pm_color_scale(), legend=alt.Legend(title="PM"))
        )
        .properties(height=280)
    )
    # smoothed overlay
    c_smooth = (
        alt.Chart(smooth.reset_index().melt("time", var_name="Series", value_name="value"))
        .mark_line(strokeDash=[4, 3])
        .encode(
            x="time:T",
            y="value:Q",
            color=alt.Color("Series:N", scale=pm_color_scale(), legend=None)
        )
    )
    st.altair_chart((c_base + c_smooth).interactive(), use_container_width=True)

    st.divider()

    # 2) Diurnal pattern
    st.subheader("🕑 Diurnal pattern (hour of day)")
    st.markdown("<div class='tip'>This shows typical within-day shape. Morning/evening peaks can suggest traffic or heating patterns.</div>", unsafe_allow_html=True)
    tmp = df.copy()
    tmp["hour"] = tmp["time"].dt.hour
    di = tmp[["hour"] + pm_cols].melt("hour", var_name="Series", value_name="value").dropna()
    ch_di = alt.Chart(di).mark_boxplot(size=18).encode(
        x=alt.X("hour:O", title="Hour"),
        y=alt.Y("value:Q", title="µg/m³"),
        color=alt.Color("Series:N", scale=pm_color_scale(), legend=alt.Legend(title="PM"))
    ).properties(height=260)
    st.altair_chart(ch_di, use_container_width=True)

    st.divider()

    # 3) Weekday pattern
    st.subheader("📅 Weekday pattern")
    st.markdown("<div class='tip'>Averaged by day. Useful to detect weekday vs weekend differences.</div>", unsafe_allow_html=True)
    tmp["weekday"] = tmp["time"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wk = tmp[["weekday"] + pm_cols].melt("weekday", var_name="Series", value_name="value").dropna()
    ch_wk = alt.Chart(wk).mark_bar().encode(
        x=alt.X("weekday:N", sort=order, title="Weekday"),
        y=alt.Y("mean(value):Q", title="Mean µg/m³"),
        color=alt.Color("Series:N", scale=pm_color_scale(), legend=alt.Legend(title="PM")),
        tooltip=[alt.Tooltip("weekday:N"), alt.Tooltip("Series:N"), alt.Tooltip("mean(value):Q", format=".1f")]
    ).properties(height=240)
    st.altair_chart(ch_wk, use_container_width=True)

    st.divider()

    # 4) PM vs Weather
    st.subheader("🌡️ PM vs weather")
    st.markdown("<div class='tip'>Weather often explains short-term changes: wind can dilute, humidity can increase PM via hygroscopic growth.</div>", unsafe_allow_html=True)
    wx_pairs = [
        ("temperature_2m", "Temperature (°C)"),
        ("relative_humidity_2m", "Relative humidity (%)"),
        ("wind_speed_10m", "Wind speed (m/s)"),
    ]
    cols = st.columns(3)
    for i, (wx, wx_label) in enumerate(wx_pairs):
        with cols[i]:
            if wx in df.columns and "pm2_5" in df.columns:
                d = df[["time", "pm2_5", wx]].dropna()
                if not d.empty:
                    d = d.assign(pm2_5_roll6=d.set_index("time")["pm2_5"].rolling("6h", min_periods=1).mean().values)
                    ch = alt.Chart(d).mark_circle(opacity=0.45, size=45).encode(
                        x=alt.X(f"{wx}:Q", title=wx_label),
                        y=alt.Y("pm2_5_roll6:Q", title="PM2.5 (6h avg)"),
                        tooltip=[alt.Tooltip(f"{wx}:Q", format=".2f"), alt.Tooltip("pm2_5_roll6:Q", format=".1f")]
                    ) + alt.Chart(d).transform_loess(wx, "pm2_5_roll6", bandwidth=0.3).mark_line()
                    st.altair_chart(ch.properties(height=240), use_container_width=True)
                else:
                    st.info(f"Not enough rows for PM2.5 vs {wx_label}.")
            else:
                st.info(f"{wx_label} not available.")

    st.divider()

    # 5) Outliers & quick stats
    st.subheader("🚨 Outliers & quick stats")
    left, right = st.columns(2)
    with left:
        st.markdown("**Summary stats**")
        show_cols = [c for c in ["pm2_5", "pm10"] if c in df.columns]
        if show_cols:
            st.dataframe(df[show_cols].describe().T, use_container_width=True)
        else:
            st.info("No PM columns to summarize.")
    with right:
        st.markdown("**Outliers (thresholds)**")
        cut_pm25 = st.number_input("PM2.5 ≥", min_value=0.0, value=150.0, step=5.0)
        cut_pm10 = st.number_input("PM10 ≥",  min_value=0.0, value=250.0, step=10.0)
        mask = pd.Series(False, index=df.index)
        if "pm2_5" in df.columns: mask |= df["pm2_5"] >= cut_pm25
        if "pm10"  in df.columns: mask |= df["pm10"]  >= cut_pm10
        out = df.loc[mask, ["time"] + [c for c in ["pm2_5", "pm10"] if c in df.columns]]
        if out.empty:
            st.success("No rows above the thresholds in this window.")
        else:
            st.dataframe(out.tail(200), use_container_width=True)

# ====================== TRAINING VIEW ======================
else:
    df, meta, err = load_training(target, limit)
    if show_debug_train:
        with st.expander("API payload (training)"):
            st.write({"meta": meta, "shape": None if df is None else df.shape, "columns": list(df.columns) if not df.empty else []})
    if err:
        st.error(err)
        st.stop()
    if df.empty:
        st.info("No data to display.")
        st.stop()

    tgt = meta.get("target", target)
    st.subheader(f"🎯 Training dataset — {tgt}")
    st.markdown("<div class='tip'>This is the feature-engineered table the model actually sees (lags, rolling stats, calendar vars, etc.).</div>", unsafe_allow_html=True)

    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"<div class='card'><div class='k'>Rows</div><div class='v'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='card'><div class='k'>Columns</div><div class='v'>{df.shape[1]:,}</div></div>", unsafe_allow_html=True)
    with k3:
        ok = (tgt in df.columns)
        st.markdown(f"<div class='card'><div class='k'>Target column present</div><div class='v'>{'Yes' if ok else 'No'}</div></div>", unsafe_allow_html=True)

    st.divider()

    # 1) Target distribution
    st.subheader("📦 Target distribution")
    st.markdown("<div class='tip'>Skew tells you if the model sees many small values and few large ones (common in PM data). Consider transforms if very skewed.</div>", unsafe_allow_html=True)
    if tgt in df.columns:
        ch = density_chart(df, tgt, f"{tgt} (µg/m³)")
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Not enough numeric data to draw target density.")
    else:
        st.warning(f"Target '{tgt}' not found in columns.")

    st.divider()

    # 2) Feature distributions (select top N by simple variance as a proxy for “interesting”)
    st.subheader("🔬 Feature distributions")
    st.markdown("<div class='tip'>Wide distributions indicate features that vary a lot and may carry signal. Narrow ones are either stable or engineered averages.</div>", unsafe_allow_html=True)
    # pick numeric features (exclude target)
    num_feats = [c for c in df.columns if c != tgt and pd.api.types.is_numeric_dtype(df[c])]
    if not num_feats:
        st.info("No numeric features available.")
    else:
        # rank by variance (high → more variation)
        var = df[num_feats].var(numeric_only=True).sort_values(ascending=False)
        topN = st.slider("Show top N features (by variance)", 6, min(24, len(var)), min(12, len(var)), step=6)
        show = var.head(topN).index.tolist()
        rows = (len(show) + 2) // 3
        for i in range(0, len(show), 3):
            c1, c2, c3 = st.columns(3)
            for j, col in enumerate(show[i:i+3]):
                with [c1, c2, c3][j]:
                    ch = density_chart(df, col, col)
                    if ch is not None:
                        st.altair_chart(ch, use_container_width=True)
                    else:
                        st.write(f"({col}: no data)")

    st.divider()

    # 3) Correlation snapshot (fast, Pearson)
    st.subheader("🔗 Correlations (Pearson)")
    st.markdown("<div class='tip'>Helps spot multicollinearity (e.g., many lagged versions highly correlated) and simple linear links to the target.</div>", unsafe_allow_html=True)
    # choose a manageable set: target + top 12 varying features to keep viz snappy
    if tgt in df.columns:
        base_cols = [tgt]
    else:
        base_cols = []
    var = df.drop(columns=[tgt], errors="ignore").var(numeric_only=True).sort_values(ascending=False)
    pick = base_cols + var.head(12).index.tolist()
    ch_corr = corr_heatmap(df, pick, "Correlation of target and top-varying features")
    if ch_corr is not None:
        st.altair_chart(ch_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    st.divider()

    # 4) Quick “families” counts (to understand engineered groups)
    st.subheader("🧩 Feature families (counts)")
    st.markdown("<div class='tip'>A quick look at how many lags/rolling stats/calendar features you have — useful sanity check for the pipeline.</div>", unsafe_allow_html=True)
    def family(name: str) -> str:
        n = name.lower()
        if n == tgt.lower(): return "Target"
        if "_lag_" in n: return "Lagged"
        if any(x in n for x in ["mean_6h","std_6h","min_6h","max_6h"]): return "Rolling 6h"
        if any(x in n for x in ["mean_12h","std_12h","min_12h","max_12h"]): return "Rolling 12h"
        if n in {"hour","weekday","month","is_weekend","hour_sin","hour_cos","wday_sin","wday_cos","month_sin","month_cos"}:
            return "Calendar/Cycle"
        if "wind_dir_" in n: return "Wind direction"
        if any(x in n for x in ["temperature_2m","relative_humidity_2m","wind_speed_10m"]): return "Weather"
        if any(x in n for x in ["pm2_5","pm10","ozone","nitrogen_dioxide","carbon_monoxide","sulphur_dioxide"]): return "Pollutants/Derived"
        return "Other"
    fam = pd.Series({c: family(c) for c in df.columns}).value_counts().rename_axis("family").reset_index(name="count")
    ch_fam = (
        alt.Chart(fam).mark_bar().encode(
            x=alt.X("family:N", sort="-y", title="Family"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=["family:N", "count:Q"]
        ).properties(height=220)
    )
    st.altair_chart(ch_fam, use_container_width=True)

# ====================== FOOTNOTE ======================
st.caption("Notes: Raw view comes from /eda/raw (Open-Meteo merge). Training view comes from /eda/training (datasets_per_target). Time-based charts only use the raw view.")
