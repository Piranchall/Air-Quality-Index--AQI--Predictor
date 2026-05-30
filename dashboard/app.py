"""
app.py — Streamlit wrapper that serves the beautiful HTML dashboard
wired to live data from the inference pipeline.

Usage:
  streamlit run dashboard/app.py
"""

import sys
import os
import json
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_pipeline.predict import run_inference
from inference_pipeline.alerts import get_alert

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide Streamlit chrome
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Cache forecast (30 min) ────────────────────────────────
@st.cache_data(ttl=1800)
def get_forecast():
    return run_inference()


# ── Load data ──────────────────────────────────────────────
with st.spinner("Loading forecast..."):
    try:
        result      = get_forecast()
        forecast    = result["forecast"]
        daily       = result["daily_summary"]
        shap_vals   = result["shap_values"]
        current_aqi = result["current_aqi"]
        model_name  = result["model"]
        metrics     = result["metrics"]
        seed_data   = result.get("seed_data", {})
        error       = None
    except Exception as e:
        error = str(e)

if error:
    st.error(f"⚠️ Failed to load forecast: {error}")
    st.stop()

# ── Build data for JS injection ────────────────────────────
alert = get_alert(current_aqi)

# Forecast values list for chart
fc_values = [round(r["predicted_aqi"]) for r in forecast]

# Day labels from daily summary
day_labels = []
for d in daily:
    try:
        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        day_labels.append(dt.strftime("%a").upper())
    except:
        day_labels.append(d["date"])

# Daily cards data
days_data = []
for d in daily:
    try:
        dt = datetime.strptime(d["date"], "%Y-%m-%d")
        name = dt.strftime("%A")
        short = dt.strftime("%a").upper()
    except:
        name = d["date"]
        short = d["date"][:3].upper()
    days_data.append({
        "name":  name,
        "short": short,
        "avg":   d["avg_aqi"],
        "min":   d["min_aqi"],
        "max":   d["max_aqi"],
    })

# SHAP data
shap_data = [
    {"f": k, "d": k.replace("_", " ").title(), "v": round(v, 2)}
    for k, v in (shap_vals.items() if shap_vals else {})
]

# Health advisory text per category
health_text = {
    "Good":                           ("No action needed", "Air quality is excellent. Enjoy outdoor activities freely."),
    "Moderate":                        ("Sensitive groups take care", "Air quality is acceptable but unusually sensitive people should consider limiting prolonged outdoor exertion."),
    "Unhealthy for Sensitive Groups":  ("Sensitive groups stay indoors", "Children, elderly and people with heart or lung conditions should reduce outdoor activity and wear a mask if going out."),
    "Unhealthy":                       ("Reduce prolonged outdoor exertion", "Air is unhealthy for everyone. Keep windows closed, run a purifier indoors, and wear an N95 if you must be outside."),
    "Very Unhealthy":                  ("Avoid all outdoor activity", "Health alert — serious effects for everyone. Stay indoors with air purification. Seal windows and doors."),
    "Hazardous":                       ("Emergency conditions — stay indoors", "Everyone must avoid outdoor exposure. This is an emergency air quality event. Wear N95 or higher at all times if forced outside."),
}
h_title, h_desc = health_text.get(alert["category"], ("Monitor conditions", "Stay aware of air quality changes."))

trend_txt = ""
if len(fc_values) >= 18:
    peak_h = fc_values.index(max(fc_values[:24]))
    peak_v = max(fc_values[:24])
    direction = "Rising" if fc_values[0] < peak_v else "Easing"
    trend_txt = f"{direction} — peaks near {peak_v} in ~{peak_h}h before easing into the weekend"
else:
    trend_txt = f"Current level: {alert['category']}"

# Current weather from seed data
pm25        = round(seed_data.get("pm25", current_aqi), 1)
temp_c      = round(seed_data.get("temp_c", 32), 1)
humidity    = round(seed_data.get("humidity_pct", 66))
wind        = round(seed_data.get("wind_speed_ms", 8.2), 1)
pressure    = round(seed_data.get("pressure_hpa", 1006))

# ── Read HTML template ─────────────────────────────────────
template_path = os.path.join(os.path.dirname(__file__), "template.html")
with open(template_path, "r", encoding="utf-8") as f:
    html = f.read()

# ── Inject live data via JS replacement ────────────────────
live_js = f"""
<script>
// ── Live data injected by Streamlit ──
window.LIVE = {{
  current:     {int(current_aqi)},
  fc:          {json.dumps(fc_values)},
  dayLabels:   {json.dumps(day_labels)},
  days:        {json.dumps(days_data)},
  shap:        {json.dumps(shap_data)},
  model:       "{model_name.replace('_', ' ').title()}",
  r2:          {metrics['r2']},
  rmse:        {metrics['rmse']},
  trendTxt:    "{trend_txt}",
  hTitle:      "{h_title}",
  hDesc:       "{h_desc}",
  pm25:        {pm25},
  tempC:       {temp_c},
  humidity:    {humidity},
  wind:        {wind},
  pressure:    {pressure},
  generatedAt: "{datetime.now(timezone.utc).strftime('%H:%M UTC')}",
}};
</script>
"""

# Patch the HTML to use live data
# 1. Inject LIVE data block before existing scripts
html = html.replace("<script>", live_js + "\n<script>", 1)

# 2. Replace static CURRENT fallback with live value
html = html.replace(
    "const CURRENT=(window.LIVE&&window.LIVE.current)||161;",
    "const CURRENT=window.LIVE.current;"
)

# 3. Replace static genForecast() with live data
html = html.replace(
    "const FC=(window.LIVE&&window.LIVE.fc)||genForecast();",
    "const FC=window.LIVE.fc;"
)

# 4. Replace static DAY_LABELS
html = html.replace(
    "const DAY_LABELS=(window.LIVE&&window.LIVE.dayLabels)||['THU','FRI','SAT'];",
    "const DAY_LABELS=window.LIVE.dayLabels;"
)

# 5. Replace static DAYS array in daily cards
html = html.replace(
    """const DAYS=(window.LIVE&&window.LIVE.days)||[
  {name:'Thursday',avg:168,min:150,max:186},
  {name:'Friday',avg:174,min:158,max:192},
  {name:'Saturday',avg:152,min:138,max:168}
];""",
    "const DAYS=window.LIVE.days;"
)

# 6. Replace static SHAP data
html = html.replace(
    "const SHAP=(window.LIVE&&window.LIVE.shap&&window.LIVE.shap.length?window.LIVE.shap:[\n  {f:'aqi_lag_1h',d:'AQI one hour ago',v:0.91},",
    "const SHAP=(window.LIVE.shap&&window.LIVE.shap.length?window.LIVE.shap:[{f:'aqi_lag_1h',d:'AQI one hour ago',v:0.91},"
)

# 8. Inject live weather readouts and model tag after page loads
live_dom_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  if (!window.LIVE) return;
  const L = window.LIVE;

  // model tag
  const mt = document.getElementById('modelTag');
  if (mt) mt.textContent = 'Model · ' + L.model + ' · 72h horizon';

  // big AQI number
  const an = document.getElementById('aqiNum');
  if (an) an.textContent = L.current;

  // weather readouts
  const readouts = document.querySelectorAll('.readout .rval');
  if (readouts.length >= 5) {
    readouts[0].innerHTML = L.pm25 + '<small>μg/m³</small>';
    // mini-bar: pm25 as % of 200 ceiling
    const bar = readouts[0].closest('.readout')?.querySelector('.mini-bar i');
    if (bar) bar.style.width = Math.min(100, Math.round(L.pm25 / 200 * 100)) + '%';
    readouts[1].innerHTML = L.tempC + '<small>°C</small>';
    readouts[2].innerHTML = L.humidity + '<small>%</small>';
    readouts[3].innerHTML = L.wind + '<small>m/s</small>';
    readouts[4].innerHTML = L.pressure + '<small>hPa</small>';
  }

  // trend text
  const tt = document.getElementById('trendTxt');
  if (tt) tt.textContent = L.trendTxt;

  // health banner
  const ht = document.getElementById('healthTitle');
  const hd = document.getElementById('healthDesc');
  const htag = document.getElementById('healthTag');
  if (ht) ht.textContent = L.hTitle;
  if (hd) hd.textContent = L.hDesc;
  if (htag) htag.textContent = 'Health Advisory · ' + document.getElementById('catWord')?.textContent;

  // generated at
  const ft = document.getElementById('footTime');
  if (ft) ft.textContent = L.generatedAt;
});
</script>
"""
html = html.replace("</body>", live_dom_js + "\n</body>")

# ── Render ─────────────────────────────────────────────────
components.html(html, height=2400, scrolling=True)