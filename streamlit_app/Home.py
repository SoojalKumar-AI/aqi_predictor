# streamlit_app/Home.py
import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

API = os.getenv("API_URL", "http://localhost:8080")
st.set_page_config(page_title="Lahore AQI Forecast", page_icon="🌫️", layout="wide")

# ---------- theme-aware styles ----------
st.markdown("""
<style>
:root { --card-bg:#ffffff; --card-border:rgba(0,0,0,.08); --text:#0f1115; --muted:rgba(0,0,0,.65); --soft:#f6f8fb; }
@media (prefers-color-scheme: dark) {
  :root { --card-bg:#17191d; --card-border:rgba(255,255,255,.14); --text:#e8eaed; --muted:rgba(255,255,255,.70); --soft:#0f1115; }
}
.main-title { font-size:1.75rem; font-weight:800; letter-spacing:.2px; }
.section-title {font-size:1.15rem; font-weight:800; margin:4px 0 10px 0}
.subtle { color:var(--muted); }
.card {
  border-radius:16px; padding:16px 18px; background:var(--card-bg);
  border:1px solid var(--card-border); box-shadow:0 10px 24px rgba(0,0,0,.06);
  color:var(--text);
}
.card.tight { padding:14px 16px; }
.k {font-weight:600;font-size:.95rem;opacity:.95}
.v {font-size:2.05rem;font-weight:800;margin-top:6px}
.b {display:inline-block;padding:3px 12px;border-radius:999px;font-size:.8rem;font-weight:700;color:#fff}
.hero {
  display:flex; align-items:center; gap:18px;
  border-radius:18px; padding:18px; border:1px solid var(--card-border);
  color:var(--text);
}
.band-outer {width:100%;height:10px;border-radius:6px;background:#3a3a3a22;overflow:hidden;margin-top:10px}
.band-inner {height:100%;border-radius:6px}
.small {font-size:.85rem;color:var(--muted)}
.legend {display:flex; gap:8px; flex-wrap:wrap}
.pill {border-radius:999px; padding:4px 10px; font-size:.8rem; font-weight:600; color:#fff;}
.block-gap {margin-top:12px}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌍 Lahore AQI — What to expect next</div>", unsafe_allow_html=True)
st.caption("Daily-updated forecasts powered by ML (with SHAP explanations under the hood).")

# ---------------- AQI helpers ----------------
def _aqi_linear(c, c_lo, c_hi, aqi_lo, aqi_hi):
    return (aqi_hi - aqi_lo) / (c_hi - c_lo) * (c - c_lo) + aqi_lo

def aqi_from_pm25(pm25):
    bps = [(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),
           (150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    for c_lo, c_hi, a_lo, a_hi in bps:
        if pm25 <= c_hi: return _aqi_linear(pm25, c_lo, c_hi, a_lo, a_hi)
    return 500.0

def aqi_from_pm10(pm10):
    bps = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),
           (355,424,201,300),(425,504,301,400),(505,604,401,500)]
    for c_lo, c_hi, a_lo, a_hi in bps:
        if pm10 <= c_hi: return _aqi_linear(pm10, c_lo, c_hi, a_lo, a_hi)
    return 500.0

ORDER = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
COLORS = {"Good":"#00A65A","Moderate":"#FFCC00","Unhealthy for Sensitive Groups":"#FF7E00",
          "Unhealthy":"#FF0000","Very Unhealthy":"#8F3F97","Hazardous":"#7E0023"}

# PM card colors
PM25_COLOR = "#2E86C1"   # blue
PM10_COLOR = "#E67E22"   # orange

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """#RRGGBB -> rgba(r,g,b,alpha) with soft transparency for hero background."""
    hex_color = hex_color.strip().lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def worst_category(c1, c2):
    if not c1: return c2
    if not c2: return c1
    return max([c1,c2], key=lambda c: ORDER.index(c))

def aqi_band(aqi):
    width = int(np.clip(aqi/500*100, 0, 100))
    if   aqi <= 50:  cat="Good"
    elif aqi <= 100: cat="Moderate"
    elif aqi <= 150: cat="Unhealthy for Sensitive Groups"
    elif aqi <= 200: cat="Unhealthy"
    elif aqi <= 300: cat="Very Unhealthy"
    else:            cat="Hazardous"
    color = COLORS.get(cat, "#6c757d")
    band_html = f"<div class='band-outer'><div class='band-inner' style='width:{width}%;background:{color}'></div></div><div class='small'>AQI {aqi:.0f} / 500</div>"
    return cat, color, band_html

def health_message(category: str) -> str:
    msg = {
        "Good": "Air quality is good. Enjoy outdoor activities.",
        "Moderate": "Acceptable for most. Very sensitive people should keep outdoor exposure shorter.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should reduce prolonged outdoor exertion.",
        "Unhealthy": "Everyone may begin to feel effects; limit outdoor activity, especially intense ones.",
        "Very Unhealthy": "Health alert. Avoid outdoor activity if possible.",
        "Hazardous": "Serious health effects likely. Stay indoors with ventilation/filtration."
    }
    return msg.get(category, "")

def card(title, value, unit, category, aqi_value=None, tint=None, tight=False):
    """Generic stat card; optional `tint` sets a soft background color."""
    badge = COLORS.get(category or "", "#6c757d")
    _, _, band_html = aqi_band(aqi_value) if aqi_value is not None else (None, None, "")
    cls = "card tight" if tight else "card"
    style_bg = f"background:{tint};" if tint else ""
    st.markdown(f"""
    <div class="{cls}" style="{style_bg}">
      <div class="k">{title}</div>
      <div class="v">{value}{' ' + unit if unit else ''}</div>
      <div style="margin:10px 0 0 0"><span class="b" style="background:{badge}">{category or '—'}</span></div>
      <div style="margin-top:10px">{band_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- API helpers ----------------
def fetch_pred(h):
    r = requests.get(f"{API}/predict", params={"horizon": h}, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_pred_payload(payload, horizon):
    k25 = "pm2_5_t+3h" if horizon==3 else "pm2_5_t+6h"
    k10 = "pm10_t+3h"  if horizon==3 else "pm10_t+6h"
    if k25 not in payload or ("error" in payload.get(k25, {})): raise RuntimeError(payload.get(k25, {}).get("error", f"Missing {k25}"))
    if k10 not in payload or ("error" in payload.get(k10, {})): raise RuntimeError(payload.get(k10, {}).get("error", f"Missing {k10}"))
    pm25=float(payload[k25]["prediction"]); cat25=payload[k25].get("category","")
    pm10=float(payload[k10]["prediction"]); cat10=payload[k10].get("category","")
    aqi25=float(aqi_from_pm25(pm25)); aqi10=float(aqi_from_pm10(pm10))
    overall_aqi = max(aqi25, aqi10)
    overall_cat, _, _ = aqi_band(overall_aqi)
    return {"pm2_5":pm25,"pm2_5_category":cat25,"pm2_5_aqi":aqi25,
            "pm10":pm10,"pm10_category":cat10,"pm10_aqi":aqi10,
            "overall_category":overall_cat,"overall_aqi":overall_aqi}

def load_ts(hours=48):
    r = requests.get(f"{API}/timeseries", params={"hours": hours}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
    return df

# ---------- Legend (uniform boxes) ----------
st.markdown("""
<style>
.legend-grid{
  display:grid; grid-template-columns:repeat(6, 1fr); gap:10px;
}
@media (max-width: 1000px){ .legend-grid{ grid-template-columns:repeat(3, 1fr);} }
@media (max-width: 640px){  .legend-grid{ grid-template-columns:repeat(2, 1fr);} }

.pill{
  border-radius:999px;
  padding:8px 12px;
  font-size:.85rem; font-weight:700; color:#fff; text-align:center;
  min-height:44px;            /* uniform height */
  display:flex; align-items:center; justify-content:center;
  line-height:1.15;           /* nicer wrapping */
  white-space:normal;         /* allow wrap for long label */
  text-wrap:balance;          /* modern browsers */
}
.section-title{font-size:1.15rem; font-weight:800; margin:4px 0 10px 0}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>Legend</div>", unsafe_allow_html=True)
labels = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
items = "".join([f"<div class='pill' style='background:{COLORS[l]}'>{l}</div>" for l in labels])
st.markdown(f"<div class='legend-grid'>{items}</div>", unsafe_allow_html=True)

st.divider()


# ---------- Data window reference (exact start → end) ----------
try:
    ts_window = load_ts(hours=168)  # one week to state the window clearly
    if not ts_window.empty:
        st.caption(f"Forecasts are based on data from **{ts_window['time'].min()}** to **{ts_window['time'].max()}**.")
    else:
        st.caption("Forecasts are based on the most recent data available.")
except Exception:
    st.caption("Forecasts are based on the most recent data available.")

# ======================================================
# Section A — “Next few hours” (≈ 3 hours from now)
# ======================================================
st.markdown("<div class='section-title'>✅ Next few hours (≈ 3 hours from now)</div>", unsafe_allow_html=True)
try:
    p3 = parse_pred_payload(fetch_pred(3), 3)
    cat3, cat3_hex, band3 = aqi_band(p3["overall_aqi"])
    hero_tint = hex_to_rgba(cat3_hex, 0.18)

    # Hero: Overall AQI with tinted background
    st.markdown(f"""
    <div class="hero" style="background:{hero_tint};">
      <div style="min-width:260px">
        <div class="k">Overall air quality</div>
        <div class="v">{cat3}</div>
        <div style="margin-top:6px"><span class="b" style="background:{cat3_hex}">{cat3}</span></div>
        <div style="margin-top:10px">{band3}</div>
      </div>
      <div class="subtle" style="font-size:0.95rem;">
        <strong>What this means:</strong><br/>{health_message(cat3)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='block-gap'></div>", unsafe_allow_html=True)

    # PM cards with meaningful fixed colors
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        card("PM2.5 forecast", f"{p3['pm2_5']:.1f}", "µg/m³",
             p3["pm2_5_category"], p3["pm2_5_aqi"], tint=hex_to_rgba(PM25_COLOR, 0.12))
    with r2c2:
        card("PM10 forecast",  f"{p3['pm10']:.1f}",  "µg/m³",
             p3["pm10_category"],  p3["pm10_aqi"],  tint=hex_to_rgba(PM10_COLOR, 0.12))
except Exception as e:
    st.error(f"Couldn’t fetch the 3-hour forecast: {e}")

st.divider()

# ======================================================
# Section B — “Later today” (≈ 6 hours from now)
# ======================================================
st.markdown("<div class='section-title'>🕒 Later today (≈ 6 hours from now)</div>", unsafe_allow_html=True)
try:
    p6 = parse_pred_payload(fetch_pred(6), 6)
    cat6, cat6_hex, band6 = aqi_band(p6["overall_aqi"])
    hero_tint6 = hex_to_rgba(cat6_hex, 0.18)

    st.markdown(f"""
    <div class="hero" style="background:{hero_tint6};">
      <div style="min-width:260px">
        <div class="k">Overall air quality</div>
        <div class="v">{cat6}</div>
        <div style="margin-top:6px"><span class="b" style="background:{cat6_hex}">{cat6}</span></div>
        <div style="margin-top:10px">{band6}</div>
      </div>
      <div class="subtle" style="font-size:0.95rem;">
        <strong>What this means:</strong><br/>{health_message(cat6)}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='block-gap'></div>", unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        card("PM2.5 forecast", f"{p6['pm2_5']:.1f}", "µg/m³",
             p6["pm2_5_category"], p6["pm2_5_aqi"], tint=hex_to_rgba(PM25_COLOR, 0.12))
    with r3c2:
        card("PM10 forecast",  f"{p6['pm10']:.1f}",  "µg/m³",
             p6["pm10_category"],  p6["pm10_aqi"],  tint=hex_to_rgba(PM10_COLOR, 0.12))
except Exception as e:
    st.error(f"Couldn’t fetch the 6-hour forecast: {e}")

st.divider()

# ======================================================
# Section C — Recent trend (UX-friendly charts)
# ======================================================
st.markdown("<div class='section-title'>📈 Last 48 hours</div>", unsafe_allow_html=True)

def pm_chart(df):
    base = alt.Chart(df).transform_fold(
        ["pm2_5","pm10"], as_=["Series","value"]
    ).mark_line().encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("value:Q", title="Concentration (µg/m³)"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["pm2_5","pm10"], range=[PM25_COLOR, PM10_COLOR]), legend=alt.Legend(title="PM")),
        tooltip=[alt.Tooltip("time:T"), alt.Tooltip("Series:N"), alt.Tooltip("value:Q", format=".1f")]
    ).properties(height=260)

    # 6-hour rolling average overlay
    roll = (df.set_index("time")[["pm2_5","pm10"]]
              .rolling("6h", min_periods=1).mean()
              .reset_index().melt("time", var_name="Series", value_name="value"))
    roll_chart = alt.Chart(roll).mark_line(strokeDash=[4,3]).encode(
        x="time:T", y="value:Q",
        color=alt.Color("Series:N", scale=alt.Scale(domain=["pm2_5","pm10"], range=[PM25_COLOR, PM10_COLOR]), legend=None)
    )
    return (base + roll_chart).interactive()

def gas_chart(df):
    cols = [c for c in ["ozone","nitrogen_dioxide","carbon_monoxide","sulphur_dioxide"] if c in df.columns]
    if not cols:
        return None
    base = alt.Chart(df).transform_fold(
        cols, as_=["Gas","value"]
    ).mark_line().encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("value:Q", title="Concentration"),
        color=alt.Color("Gas:N", legend=alt.Legend(title="Gases")),
        tooltip=[alt.Tooltip("time:T"), alt.Tooltip("Gas:N"), alt.Tooltip("value:Q", format=".1f")]
    ).properties(height=260)
    return base.interactive()

try:
    ts48 = load_ts(hours=48)
    if not ts48.empty:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**PM particles**")
            st.altair_chart(pm_chart(ts48), use_container_width=True)
        with c_right:
            st.markdown("**Gases**")
            g = gas_chart(ts48)
            if g is not None:
                st.altair_chart(g, use_container_width=True)
            else:
                st.info("No pollutant breakdown available from the server.")
        st.caption(f"Window shown: **{ts48['time'].min()} → {ts48['time'].max()}**")
    else:
        st.info("No recent time series available.")
except Exception as e:
    st.error(f"Failed to load time series: {e}")
