# streamlit_app/pages/01_Forecast.py
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import timedelta

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Forecast", page_icon="🔮", layout="wide")

# =========================
# THEME & STYLES
# =========================
st.markdown("""
<style>
:root { --card-bg:#ffffff; --card-border:rgba(0,0,0,.10); --text:#0f1115; --muted:rgba(0,0,0,.65); --soft:#f6f8fb; }
@media (prefers-color-scheme: dark) {
  :root { --card-bg:#17191d; --card-border:rgba(255,255,255,.14); --text:#e8eaed; --muted:rgba(255,255,255,.70); --soft:#0f1115; }
}
/* force text contrast by theme */
body, .title, .section, .subtle, .kpi-card, .kpi-card *:not(.badge), .hero, .hero *:not(.badge),
.k, .v, .small, .pill, .legend-label {
  color:#000 !important;
}
@media (prefers-color-scheme: dark) {
  body, .title, .section, .subtle, .kpi-card, .kpi-card *:not(.badge), .hero, .hero *:not(.badge),
  .k, .v, .small, .pill, .legend-label {
    color:#fff !important;
  }
}
/* layout bits */
.title { font-size:1.8rem; font-weight:800; letter-spacing:.2px; margin-bottom:2px; }
.tagline { color:var(--muted); margin-bottom:14px; }
.section { font-size:1.1rem; font-weight:800; margin:4px 0 12px 0; position:sticky; top:0; backdrop-filter:blur(6px); padding-top:8px; z-index:5;}
.kpi-card {
  border-radius:16px; padding:16px 18px; background:var(--card-bg);
  border:1px solid var(--card-border); box-shadow:0 10px 22px rgba(0,0,0,.06);
}
.k {font-weight:600; font-size:.95rem; opacity:.95}
.v {font-size:2.05rem; font-weight:800; margin-top:6px}
.badge {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.8rem; font-weight:700; color:#fff}
.band-outer {width:100%; height:10px; border-radius:6px; background:#3a3a3a22; overflow:hidden; margin-top:10px}
.band-inner {height:100%; border-radius:6px}
.small {font-size:.85rem; color:var(--muted)}
.hero {
  display:flex; gap:18px; align-items:flex-start;
  border-radius:18px; padding:18px; border:1px solid var(--card-border);
}
.hero-col { flex: 0 0 290px; }
.hero-tip { font-size:.95rem; opacity:.95 }
.legend {display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 10px 0}
.pill {border-radius:999px; padding:4px 10px; font-size:.8rem; font-weight:600; color:#fff;}
.soft { background: var(--soft); border-radius:12px; padding:10px 12px; border:1px solid var(--card-border); }
.download-box { text-align:right; }
hr { border: none; border-top: 1px solid var(--card-border); margin: 10px 0 14px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔮 Detailed Forecast</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Clear outlooks, AQI bands, health tips, and top feature drivers</div>", unsafe_allow_html=True)

# =========================
# CONSTANTS & HELPERS
# =========================
ORDER = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
COLORS = {
    "Good":"#00A65A","Moderate":"#FFCC00","Unhealthy for Sensitive Groups":"#FF7E00",
    "Unhealthy":"#FF0000","Very Unhealthy":"#8F3F97","Hazardous":"#7E0023"
}
PM25_COLOR = "#2E86C1"   # blue for PM2.5
PM10_COLOR = "#E67E22"   # orange for PM10

PM25_BPS = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BPS = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]

def hex_rgba(hex_color: str, alpha: float = 0.16) -> str:
    hex_color = hex_color.strip().lstrip("#")
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def cat_color(c): return COLORS.get(c, "#6c757d")

def worst_cat(c1,c2):
    if not c1: return c2
    if not c2: return c1
    return max([c1,c2], key=lambda c: ORDER.index(c))

def aqi_from_conc(c, bps):
    c = float(c)
    for Cl,Ch,Il,Ih in bps:
        if Cl <= c <= Ch:
            return (Ih-Il)/(Ch-Cl)*(c-Cl)+Il
    return min(500.0, max(0.0, c))

def aqi25(c): return aqi_from_conc(c, PM25_BPS)
def aqi10(c): return aqi_from_conc(c, PM10_BPS)
def cat_for_aqi(a):
    return "Good" if a<=50 else ("Moderate" if a<=100 else ("Unhealthy for Sensitive Groups" if a<=150 else ("Unhealthy" if a<=200 else ("Very Unhealthy" if a<=300 else "Hazardous"))))

def band_html(aqi):
    width=int(np.clip(aqi/500*100,0,100)); color=cat_color(cat_for_aqi(aqi))
    return f"<div class='band-outer'><div class='band-inner' style='width:{width}%;background:{color}'></div></div><div class='small'>AQI {aqi:.0f} / 500</div>"

def health_message(category: str) -> str:
    msg = {
        "Good": "Air quality is good. Enjoy outdoor activities.",
        "Moderate": "Acceptable for most. Very sensitive people should keep outdoor exposure shorter.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should reduce prolonged outdoor exertion.",
        "Unhealthy": "Everyone may begin to feel effects; limit outdoor activity, especially intense ones.",
        "Very Unhealthy": "Health alert. Avoid outdoor activity if possible.",
        "Hazardous": "Serious health effects likely. Stay indoors with filtration if available."
    }
    return msg.get(category, "")

def kpi_card(title, value, unit, category, foot_html="", tint=None):
    bg = f"background:{tint};" if tint else ""
    st.markdown(f"""
    <div class='kpi-card' style="{bg}">
      <div class='k'>{title}</div>
      <div class='v'>{value}{(' ' + unit) if unit else ''}</div>
      <div style="margin:8px 0 0 0"><span class='badge' style='background:{cat_color(category)}'>{category or '—'}</span></div>
      <div style="margin-top:10px">{foot_html}</div>
    </div>
    """, unsafe_allow_html=True)

def parse_pred_payload(payload, horizon):
    k25 = "pm2_5_t+3h" if horizon==3 else "pm2_5_t+6h"
    k10 = "pm10_t+3h"  if horizon==3 else "pm10_t+6h"
    if k25 not in payload or ("error" in payload.get(k25, {})): raise RuntimeError(payload.get(k25, {}).get("error", f"Missing {k25}"))
    if k10 not in payload or ("error" in payload.get(k10, {})): raise RuntimeError(payload.get(k10, {}).get("error", f"Missing {k10}"))
    pm25=float(payload[k25]["prediction"]); cat25=payload[k25].get("category","")
    pm10=float(payload[k10]["prediction"]); cat10=payload[k10].get("category","")
    a25, a10 = aqi25(pm25), aqi10(pm10)
    overall_aqi = max(a25, a10)
    overall_cat = cat_for_aqi(overall_aqi)
    return {
        "pm2_5": pm25, "pm2_5_category": cat25, "pm2_5_aqi": a25,
        "pm10": pm10, "pm10_category": cat10, "pm10_aqi": a10,
        "overall_category": overall_cat, "overall_aqi": overall_aqi
    }

def fetch_explanations(target, k=12):
    try:
        r = requests.get(f"{API}/explanations", params={"target":target,"top_k":k}, timeout=20)
        r.raise_for_status()
        feats = r.json().get("features", [])
        return pd.DataFrame(feats) if feats else None
    except Exception:
        return None

def shap_bar(df, title):
    if df is None or df.empty:
        st.info("No explanation data available.")
        return
    # try to detect a SHAP magnitude column
    cand_cols = [c for c in df.columns if "shap" in c.lower() or "value" in c.lower()]
    value_col = cand_cols[0] if cand_cols else (df.columns[1] if len(df.columns)>1 else None)
    if value_col is None:
        st.info("No SHAP values in the response.")
        return
    # normalize column names
    if "feature" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature"})
    chart = (alt.Chart(df)
             .transform_window(rank="rank()", sort=[alt.SortField(value_col, order="descending")])
             .transform_filter(alt.datum.rank <= 12)
             .mark_bar()
             .encode(
                x=alt.X(f"{value_col}:Q", title="Mean |SHAP|"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                tooltip=["feature", alt.Tooltip(f"{value_col}:Q", format=".4f")]
             )
             .properties(height=290, title=title))
    st.altair_chart(chart, use_container_width=True)

def load_ts(hours=168):
    r = requests.get(f"{API}/timeseries", params={"hours": hours}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
    return df

def last_deltas(ts: pd.DataFrame):
    """Return 24h change for PM2.5/PM10 (mean last 6h vs mean previous 6h) for trend badges."""
    if ts.empty or not set(["pm2_5","pm10"]).issubset(ts.columns):
        return None, None
    tmax = ts["time"].max()
    win2 = ts[(ts["time"] > tmax - timedelta(hours=6))]
    win1 = ts[(ts["time"] > tmax - timedelta(hours=12)) & (ts["time"] <= tmax - timedelta(hours=6))]
    def safe_mean(s): return float(s.dropna().mean()) if not s.dropna().empty else np.nan
    d25 = safe_mean(win2["pm2_5"]) - safe_mean(win1["pm2_5"])
    d10 = safe_mean(win2["pm10"])  - safe_mean(win1["pm10"])
    return d25, d10

# =========================
# HEADER: Legend + Window
# =========================
with st.container():
    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown("<div class='section'>Legend</div>", unsafe_allow_html=True)
        labels = ORDER
        st.markdown(
            "<div class='legend'>" + "".join(
                [f"<span class='pill' style='background:{COLORS[l]}'>{l}</span>" for l in labels]
            ) + "</div>", unsafe_allow_html=True
        )
    with cols[1]:
        # data window
        try:
            ts_window = load_ts(hours=168)
            if not ts_window.empty:
                st.caption(f"Forecasts are based on data from **{ts_window['time'].min()}** to **{ts_window['time'].max()}**.")
            else:
                st.caption("Forecasts are based on the most recent data available.")
        except Exception:
            st.caption("Forecasts are based on the most recent data available.")

st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# SECTIONS (3h & 6h)
# =========================
SECTIONS = [("✅ Next few hours (≈ 3 hours from now)", 3),
            ("🕒 Later today (≈ 6 hours from now)", 6)]

for label, h in SECTIONS:
    st.markdown(f"<div class='section'>{label}</div>", unsafe_allow_html=True)

    # fetch forecast & ts in parallel spirit (sequential here)
    try:
        rpred = requests.get(f"{API}/predict", params={"horizon": h}, timeout=60)
        rpred.raise_for_status()
        res = parse_pred_payload(rpred.json(), h)
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        continue

    # hero banner
    cat = res["overall_category"]
    cat_hex = COLORS.get(cat, "#6c757d")
    hero_tint = hex_rgba(cat_hex, 0.18)
    st.markdown(f"""
    <div class="hero" style="background:{hero_tint};">
      <div class="hero-col">
        <div class="k">Overall air quality</div>
        <div class="v">{cat}</div>
        <div style="margin-top:6px"><span class="badge" style="background:{cat_hex}">{cat}</span></div>
        <div style="margin-top:10px">{band_html(res["overall_aqi"])}</div>
      </div>
      <div class="hero-tip">
        <strong>What this means:</strong><br/>{health_message(cat)}
        <div class="small" style="margin-top:10px;">
          This outlook blends PM2.5 and PM10 by the worst category to keep guidance conservative.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # side-by-side KPI cards
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        kpi_card("PM2.5 forecast", f"{res['pm2_5']:.1f}", "µg/m³",
                 res["pm2_5_category"], band_html(res["pm2_5_aqi"]),
                 tint=hex_rgba(PM25_COLOR, 0.12))
    with c2:
        kpi_card("PM10 forecast", f"{res['pm10']:.1f}", "µg/m³",
                 res["pm10_category"], band_html(res["pm10_aqi"]),
                 tint=hex_rgba(PM10_COLOR, 0.12))
    with c3:
        try:
            ts48 = load_ts(hours=48)
            d25, d10 = last_deltas(ts48)

            def show_trend(label, d):
                if d is None or np.isnan(d):
                    st.metric(label, "—", delta="no change", delta_color="off")
                    return
                up = d > 0
                value = f"{abs(d):.1f} µg/m³"
                # Make rising (worse) appear red, falling (better) green
                delta_color = "inverse" if up else "normal"
                arrow = "↑" if up else ("↓" if d < 0 else "⟲")
                st.metric(label, value, delta=f"{arrow} {'rising' if up else ('falling' if d < 0 else 'flat')} vs prev 6h",
                        delta_color=delta_color)

            st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
            st.markdown("<div class='k'>What's trending</div>", unsafe_allow_html=True)
            show_trend("PM2.5 change", d25)
            show_trend("PM10 change",  d10)
            st.markdown("<div class='small'>Simple 6h vs previous 6h averages for quick direction.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            st.info("Trend data unavailable.")

    # SHAP explanations
    st.markdown("<br/>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Top drivers — PM2.5", "Top drivers — PM10"])
    tgt25 = "pm2_5_t+3h" if h==3 else "pm2_5_t+6h"
    tgt10 = "pm10_t+3h"  if h==3 else "pm10_t+6h"
    with t1:
        shap_df_25 = fetch_explanations(tgt25, 12)
        shap_bar(shap_df_25, f"{tgt25} — feature impact")
        if shap_df_25 is not None and not shap_df_25.empty:
            st.download_button("Download PM2.5 SHAP CSV",
                               data=shap_df_25.to_csv(index=False).encode("utf-8"),
                               file_name=f"{tgt25}_shap_top12.csv",
                               mime="text/csv")
    with t2:
        shap_df_10 = fetch_explanations(tgt10, 12)
        shap_bar(shap_df_10, f"{tgt10} — feature impact")
        if shap_df_10 is not None and not shap_df_10.empty:
            st.download_button("Download PM10 SHAP CSV",
                               data=shap_df_10.to_csv(index=False).encode("utf-8"),
                               file_name=f"{tgt10}_shap_top12.csv",
                               mime="text/csv")

    st.markdown("<hr/>", unsafe_allow_html=True)

# =========================
# OPTIONAL: Compact PM Charts (last 48h)
# =========================
st.markdown("<div class='section'>📈 Recent trend (last 48 hours)</div>", unsafe_allow_html=True)
try:
    ts48 = load_ts(hours=48)
    if not ts48.empty:
        # PM chart
        base = alt.Chart(ts48).transform_fold(
            ["pm2_5","pm10"], as_=["Series","value"]
        ).mark_line().encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("value:Q", title="Concentration (µg/m³)"),
            color=alt.Color("Series:N",
                            scale=alt.Scale(domain=["pm2_5","pm10"], range=[PM25_COLOR, PM10_COLOR]),
                            legend=alt.Legend(title="PM")),
            tooltip=[alt.Tooltip("time:T"), alt.Tooltip("Series:N"), alt.Tooltip("value:Q", format=".1f")]
        ).properties(height=280)

        # rolling overlay
        roll = (ts48.set_index("time")[["pm2_5","pm10"]]
                  .rolling("6h", min_periods=1).mean()
                  .reset_index().melt("time", var_name="Series", value_name="value"))
        overlay = alt.Chart(roll).mark_line(strokeDash=[4,3]).encode(
            x="time:T", y="value:Q",
            color=alt.Color("Series:N",
                            scale=alt.Scale(domain=["pm2_5","pm10"], range=[PM25_COLOR, PM10_COLOR]),
                            legend=None)
        )
        st.altair_chart((base + overlay).interactive(), use_container_width=True)

        st.caption(f"Window shown: **{ts48['time'].min()} → {ts48['time'].max()}**")
        st.markdown("<div class='download-box'>", unsafe_allow_html=True)
        st.download_button("Download last 48h CSV", data=ts48.to_csv(index=False).encode("utf-8"),
                           file_name="lahore_last_48h.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No recent time series available.")
except Exception as e:
    st.error(f"Failed to load time series: {e}")
