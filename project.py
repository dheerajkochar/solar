# ============================================================
# SOLAR OPTIMIZATION DASHBOARD — Streamlit App
# ============================================================
# Run with:  streamlit run solar_dashboard.py
# Install:   pip install streamlit pandas numpy scikit-learn plotly requests
# ============================================================

import os
import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="☀️ Solar Panel Optimizer",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Dark solar-tech aesthetic
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1e35 0%, #0a1628 100%);
    border-right: 1px solid rgba(251, 191, 36, 0.2);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(251,191,36,0.08) 0%, rgba(15,30,53,0.9) 100%);
    border: 1px solid rgba(251, 191, 36, 0.25);
    border-radius: 12px;
    padding: 16px !important;
    backdrop-filter: blur(10px);
}

[data-testid="metric-container"] label {
    color: #fbbf24 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f9ff !important;
    font-size: 1.9rem !important;
    font-weight: 700;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

/* Section header divider */
.section-header {
    font-family: 'Space Mono', monospace;
    color: #fbbf24;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(251, 191, 36, 0.3);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Recommendation cards */
.rec-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(15,30,53,0.8) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.92rem;
    color: #d1fae5;
}

.rec-card.warn {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08) 0%, rgba(15,30,53,0.8) 100%);
    border-color: rgba(251, 191, 36, 0.3);
    color: #fef3c7;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    color: #94a3b8;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #fbbf24 !important;
    border-bottom-color: #fbbf24 !important;
}

/* Sliders */
[data-testid="stSlider"] .stSlider > div > div > div {
    background: #fbbf24 !important;
}

/* Selectbox & number input */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(15, 30, 53, 0.9) !important;
    border: 1px solid rgba(251, 191, 36, 0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: #0a0e1a;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(251, 191, 36, 0.4);
}

/* Info/success boxes */
.stAlert {
    background: rgba(15, 30, 53, 0.8) !important;
    border-radius: 10px !important;
}

/* Plotly chart backgrounds should be transparent */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Hero badge */
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(251,191,36,0.15), rgba(251,191,36,0.05));
    border: 1px solid rgba(251,191,36,0.4);
    border-radius: 50px;
    padding: 4px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #fbbf24;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

PVWATTS_URL     = "https://developer.nrel.gov/api/pvwatts/v8.json"
DATASET_PATH    = "solar_dataset.csv"
TILT_VALUES     = [10, 15, 20, 25, 30, 35, 40]
LOSS_VALUES     = [8, 10, 12, 14, 16, 18]
CAPACITY_VALUES = [1, 2, 3]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,18,40,0.6)",
    font=dict(family="Sora, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=50, b=40),
)

# Reusable axis style — apply individually per chart to avoid keyword conflicts
AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ============================================================
# CACHED DATA & MODEL FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def fetch_from_api(api_key: str, lat: float, lon: float) -> pd.DataFrame:
    records = []
    total   = len(TILT_VALUES) * len(LOSS_VALUES) * len(CAPACITY_VALUES)
    done    = 0
    prog    = st.progress(0, text="Fetching from PVWatts API …")

    for tilt in TILT_VALUES:
        for loss in LOSS_VALUES:
            for capacity in CAPACITY_VALUES:
                params = dict(
                    api_key=api_key, lat=lat, lon=lon,
                    system_capacity=capacity, azimuth=180,
                    tilt=tilt, array_type=1, module_type=0, losses=loss,
                )
                try:
                    resp    = requests.get(PVWATTS_URL, params=params, timeout=15)
                    data    = resp.json()
                    outputs = data.get("outputs", {})
                    ac      = outputs.get("ac_monthly", [])
                    sol     = outputs.get("solrad_monthly", [])
                    if len(ac) == 12 and len(sol) == 12:
                        for m in range(12):
                            records.append(dict(
                                month=m+1, tilt_angle=tilt,
                                system_loss=loss, capacity_kw=capacity,
                                power_output_w=ac[m],
                                solar_radiation_kwh_m2_day=sol[m],
                            ))
                except Exception:
                    pass
                done += 1
                prog.progress(done / total, text=f"Fetching … {done}/{total}")
                time.sleep(0.4)

    prog.empty()
    df = pd.DataFrame(records)
    df.to_csv(DATASET_PATH, index=False)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["normalized_power"] = df["power_output_w"] / df["capacity_kw"]
    df["efficiency"]       = df["power_output_w"] / df["solar_radiation_kwh_m2_day"].replace(0, np.nan)
    return df.dropna(subset=["efficiency"])


# month intentionally excluded — it is correlated with radiation in training data,
# causing the model to learn month "personality" separately from radiation.
# Radiation alone encodes seasonality: same radiation → same prediction (physically honest).
FEATURES = ["tilt_angle", "system_loss", "solar_radiation_kwh_m2_day"]

@st.cache_resource(show_spinner=False)
def train_models(df_hash: int, _df: pd.DataFrame):
    """Train power & efficiency models. df_hash used only as cache key."""
    df = engineer_features(_df)

    def _train(target):
        X = df[FEATURES]; y = df[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        m = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        yp   = m.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, yp))
        r2   = r2_score(yte, yp)
        return m, rmse, r2

    mp, rmse_p, r2_p = _train("normalized_power")
    me, _,      r2_e = _train("efficiency")   # rmse_e excluded — not displayed
    return mp, me, rmse_p, r2_p, r2_e


def find_optimal_tilt(model, radiation, loss, tilt_range=range(10, 45)):
    rows = pd.DataFrame([
        {"tilt_angle": t, "system_loss": loss,
         "solar_radiation_kwh_m2_day": radiation}
        for t in tilt_range
    ])
    preds = model.predict(rows[FEATURES])
    idx   = np.argmax(preds)
    return list(tilt_range)[idx], preds[idx], list(tilt_range), preds.tolist()


def get_recommendations(radiation, loss, optimal_tilt):
    recs = []
    if loss > 12:
        recs.append(("warn", "⚠ System loss > 12% — inspect wiring, connectors, and inverter."))
    else:
        recs.append(("ok", "✓ System losses are within acceptable range."))
    if radiation < 5.0:
        recs.append(("warn", "⚠ Low irradiance — consider seasonal shading analysis."))
    else:
        recs.append(("ok", f"✓ Solar irradiance of {radiation} kWh/m²/day is healthy."))
    if optimal_tilt > 30:
        recs.append(("warn", f"↑ Increase tilt to {optimal_tilt}° for better annual yield."))
    else:
        recs.append(("ok", f"✓ Tilt of {optimal_tilt}° is near-optimal for this location."))
    return recs


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="hero-badge">☀ Solar AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Configuration")

    api_key = st.text_input(
        "NREL API Key",
        value=os.getenv("NREL_API_KEY", ""),
        type="password",
        help="Get a free key at developer.nrel.gov",
    )

    st.markdown("**Location**")
    col_lat, col_lon = st.columns(2)
    lat = col_lat.number_input("Latitude",  value=21.15, format="%.4f")
    lon = col_lon.number_input("Longitude", value=75.72, format="%.4f")

    st.divider()
    st.markdown('<p class="section-header">Scenario Parameters</p>', unsafe_allow_html=True)
    st.caption("Month excluded from model — radiation alone encodes seasonality.")

    radiation = st.slider("Solar Radiation (kWh/m²/day)", 3.0, 9.0, 6.5, 0.1)
    loss      = st.slider("System Loss (%)",              5,   25,  10,  1)
    capacity  = st.select_slider("Capacity (kW)", options=[1, 2, 3, 4, 5], value=3)

    st.divider()
    fetch_btn = st.button("🔄 Fetch Fresh Data from API", use_container_width=True)


# ============================================================
# DATA LOADING
# ============================================================

df_raw = load_dataset(DATASET_PATH)

if fetch_btn:
    if not api_key:
        st.sidebar.error("Please enter your NREL API key first.")
    else:
        with st.spinner("Calling PVWatts API …"):
            df_raw = fetch_from_api(api_key, lat, lon)
        st.sidebar.success(f"✓ Fetched {len(df_raw):,} rows")
        st.cache_resource.clear()

# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div class="hero-badge">ML-Powered · PVWatts API · Random Forest</div>
    <h1 style="font-size: 2.4rem; margin: 0; color: #f0f9ff; letter-spacing: -0.02em;">
        Solar Panel <span style="color: #fbbf24;">Optimizer</span>
    </h1>
    <p style="color: #64748b; font-size: 1rem; margin-top: 6px; font-family: Sora, sans-serif;">
        Tilt angle optimization · Efficiency prediction · Multi-scenario analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NO DATA STATE
# ============================================================

if df_raw is None or df_raw.empty:
    st.info(
        "No dataset found. Enter your NREL API key in the sidebar and click "
        "**🔄 Fetch Fresh Data from API** to get started.",
        icon="📡",
    )
    st.stop()

# ============================================================
# TRAIN MODELS
# ============================================================

with st.spinner("Training ML models …"):
    df_hash = hash(str(df_raw.shape) + str(df_raw.columns.tolist()))
    model_power, model_eff, rmse_p, r2_p, r2_e = train_models(df_hash, df_raw)

df = engineer_features(df_raw)

# ============================================================
# TOP METRICS ROW
# ============================================================

best_tilt_p, _, tilts_p, preds_p = find_optimal_tilt(model_power, radiation, loss)
best_tilt_e, _, tilts_e, preds_e = find_optimal_tilt(model_eff,   radiation, loss)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Dataset Rows",       f"{len(df_raw):,}")
m2.metric("Power Model R²",     f"{r2_p:.3f}")
m3.metric("Efficiency Model R²",f"{r2_e:.3f}")
m4.metric("Optimal Tilt (Power)",    f"{best_tilt_p}°")
m5.metric("Optimal Tilt (Efficiency)",f"{best_tilt_e}°")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯  Tilt Optimizer",
    "📊  Data Explorer",
    "🤖  Model Insights",
    "🔮  Scenario Analysis",
])


# ─────────────────────────────────────────────────────────────
# TAB 1 — TILT OPTIMIZER
# ─────────────────────────────────────────────────────────────

with tab1:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<p class="section-header">Tilt Angle vs. Predicted Output</p>', unsafe_allow_html=True)

        fig = go.Figure()

        # Power trace
        fig.add_trace(go.Scatter(
            x=tilts_p, y=preds_p,
            mode="lines+markers",
            name="Normalised Power (W/kW)",
            line=dict(color="#fbbf24", width=2.5),
            marker=dict(size=5),
            yaxis="y1",
        ))

        # Efficiency trace on secondary axis
        fig.add_trace(go.Scatter(
            x=tilts_e, y=preds_e,
            mode="lines+markers",
            name="Efficiency",
            line=dict(color="#34d399", width=2.5, dash="dot"),
            marker=dict(size=5),
            yaxis="y2",
        ))

        # Optimal tilt vertical lines
        fig.add_vline(x=best_tilt_p, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"Best {best_tilt_p}°", annotation_font_color="#fbbf24")
        fig.add_vline(x=best_tilt_e, line_dash="dash", line_color="#34d399",
                      annotation_text=f"  Best {best_tilt_e}°", annotation_font_color="#34d399",
                      annotation_position="bottom right")

        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            yaxis=dict(title="Normalised Power (W/kW)", color="#fbbf24",
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Efficiency", color="#34d399", overlaying="y", side="right",
                        gridcolor="rgba(255,255,255,0)"),
            xaxis=dict(title="Tilt Angle (°)", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">System Recommendations</p>', unsafe_allow_html=True)

        recs = get_recommendations(radiation, loss, best_tilt_e)
        for kind, text in recs:
            css_class = "rec-card" if kind == "ok" else "rec-card warn"
            st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header" style="margin-top:24px;">Optimal Config</p>',
                    unsafe_allow_html=True)

        cfg = pd.DataFrame({
            "Parameter":  ["Tilt Angle", "System Loss", "Radiation", "Capacity"],
            "Value":      [f"{best_tilt_e}°", f"{loss}%",
                           f"{radiation} kWh/m²/day", f"{capacity} kW"],
        })
        st.dataframe(cfg, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 2 — DATA EXPLORER
# ─────────────────────────────────────────────────────────────

with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="section-header">Monthly Power Output by Tilt</p>',
                    unsafe_allow_html=True)
        df_cap = df[df["capacity_kw"] == capacity].copy()
        df_pivot = df_cap.groupby(["month", "tilt_angle"])["normalized_power"].mean().reset_index()

        fig2 = px.line(
            df_pivot, x="month", y="normalized_power",
            color="tilt_angle",
            labels={"month": "Month", "normalized_power": "W/kW", "tilt_angle": "Tilt (°)"},
            color_discrete_sequence=px.colors.sequential.YlOrBr,
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=340)
        fig2.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_NAMES, **AXIS_STYLE)
        fig2.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Power vs Solar Radiation (scatter)</p>',
                    unsafe_allow_html=True)
        sample = df.sample(min(2000, len(df)), random_state=1)
        fig3 = px.scatter(
            sample, x="solar_radiation_kwh_m2_day", y="normalized_power",
            color="tilt_angle", size_max=6,
            labels={"solar_radiation_kwh_m2_day": "Solar Radiation (kWh/m²/day)",
                    "normalized_power": "Normalised Power (W/kW)", "tilt_angle": "Tilt (°)"},
            color_continuous_scale="YlOrBr",
        )
        fig3.update_layout(**PLOTLY_LAYOUT, height=340,
                           coloraxis_colorbar=dict(title="Tilt (°)"))
        fig3.update_traces(marker=dict(opacity=0.6, size=4))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section-header">Monthly Average Solar Radiation Heatmap</p>',
                unsafe_allow_html=True)
    heat = df.groupby(["month", "tilt_angle"])["solar_radiation_kwh_m2_day"].mean().reset_index()
    heat_pivot = heat.pivot(index="tilt_angle", columns="month", values="solar_radiation_kwh_m2_day")
    heat_pivot.columns = MONTH_NAMES

    fig4 = px.imshow(
        heat_pivot,
        color_continuous_scale="YlOrBr",
        labels=dict(x="Month", y="Tilt Angle (°)", color="kWh/m²/day"),
        aspect="auto",
    )
    fig4.update_layout(**PLOTLY_LAYOUT, height=280,
                       coloraxis_colorbar=dict(title="kWh/m²/day"))
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📋 Raw Dataset Preview"):
        st.dataframe(df_raw.head(200), use_container_width=True)
        st.caption(f"Showing 200 of {len(df_raw):,} rows")


# ─────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────

with tab3:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="section-header">Feature Importance — Power Model</p>',
                    unsafe_allow_html=True)

        imp_p = pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": model_power.feature_importances_,
        }).sort_values("Importance")

        fig5 = px.bar(
            imp_p, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="YlOrBr",
        )
        fig5.update_layout(**PLOTLY_LAYOUT, height=280,
                           showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

        st.metric("Power Model RMSE", f"{rmse_p:.2f}")
        st.metric("Power Model R²",   f"{r2_p:.4f}")

    with c2:
        st.markdown('<p class="section-header">Feature Importance — Efficiency Model</p>',
                    unsafe_allow_html=True)

        imp_e = pd.DataFrame({
            "Feature":    FEATURES,
            "Importance": model_eff.feature_importances_,
        }).sort_values("Importance")

        fig6 = px.bar(
            imp_e, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Teal",
        )
        fig6.update_layout(**PLOTLY_LAYOUT, height=280,
                           showlegend=False,
                           coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)

        st.metric("Efficiency Model R²",   f"{r2_e:.4f}")

    # Loss vs Power heatmap
    st.markdown('<p class="section-header">Predicted Power — Tilt × Loss Heatmap</p>',
                unsafe_allow_html=True)

    tilt_grid = list(range(10, 45, 2))
    loss_grid = list(range(8, 20, 2))
    rows = []
    for t in tilt_grid:
        for l in loss_grid:
            row = {"tilt_angle": t, "system_loss": l,
                   "solar_radiation_kwh_m2_day": radiation}
            rows.append(row)

    grid_df  = pd.DataFrame(rows)
    grid_df["pred"] = model_power.predict(grid_df[FEATURES])
    heat2    = grid_df.pivot(index="tilt_angle", columns="system_loss", values="pred")

    fig7 = px.imshow(
        heat2,
        color_continuous_scale="YlOrBr",
        labels=dict(x="System Loss (%)", y="Tilt Angle (°)", color="W/kW"),
        aspect="auto",
    )
    fig7.update_layout(**PLOTLY_LAYOUT, height=320,
                       coloraxis_colorbar=dict(title="W/kW"))
    st.plotly_chart(fig7, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 4 — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────

with tab4:
    st.markdown('<p class="section-header">Run Multiple Scenarios</p>', unsafe_allow_html=True)

    st.markdown("Define up to 5 custom scenarios and compare optimal tilt angles and efficiency side-by-side.")

    default_scenarios = [
        {"Radiation": 5.5, "Loss": 12, "Month": 1,  "Label": "Jan — Low Sun"},
        {"Radiation": 6.5, "Loss": 10, "Month": 5,  "Label": "May — Pre-Monsoon"},
        {"Radiation": 7.2, "Loss":  8, "Month": 8,  "Label": "Aug — Peak Summer"},
    ]

    n_scenarios = st.number_input("Number of scenarios", 1, 5, 3)

    scenario_inputs = []
    cols = st.columns(int(n_scenarios), gap="small")
    for i, col in enumerate(cols):
        with col:
            default = default_scenarios[i] if i < len(default_scenarios) else \
                      {"Radiation": 6.0, "Loss": 10, "Month": 6, "Label": f"Scenario {i+1}"}
            st.markdown(f"**Scenario {i+1}**")
            label = st.text_input("Label", default["Label"], key=f"lbl_{i}")
            rad   = st.number_input("Radiation", 3.0, 9.0, float(default["Radiation"]),
                                    0.1, key=f"rad_{i}")
            ls    = st.number_input("Loss (%)", 5, 25, default["Loss"], 1, key=f"los_{i}")
            mo    = st.selectbox("Month", list(range(1, 13)),
                                 index=default["Month"]-1,
                                 format_func=lambda m: MONTH_NAMES[m-1],
                                 key=f"mo_{i}")
            scenario_inputs.append({"label": label, "radiation": rad, "loss": ls, "month": mo})

    if st.button("▶ Run Scenario Analysis", use_container_width=True):
        results = []
        for sc in scenario_inputs:
            tilt_p, val_p, _, _ = find_optimal_tilt(model_power, sc["radiation"], sc["loss"])
            tilt_e, val_e, _, _ = find_optimal_tilt(model_eff,   sc["radiation"], sc["loss"])
            results.append({
                "Scenario":       sc["label"],
                "Month":          MONTH_NAMES[sc["month"]-1],
                "Radiation":      sc["radiation"],
                "Loss (%)":       sc["loss"],
                "Optimal Tilt (Power)": tilt_p,
                "Optimal Tilt (Eff.)":  tilt_e,
                "Pred. Power (W/kW)":   round(val_p, 2),
                "Pred. Efficiency":     round(val_e, 4),
            })
        st.session_state["scenario_results"]  = results
        st.session_state["scenario_inputs_ss"] = scenario_inputs

    if "scenario_results" in st.session_state:
        results       = st.session_state["scenario_results"]
        _sc_inputs    = st.session_state.get("scenario_inputs_ss", scenario_inputs)
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        fig8 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Optimal Tilt Angle (°)", "Predicted Efficiency"],
                             horizontal_spacing=0.12)
        colors = ["#fbbf24", "#34d399", "#60a5fa", "#f472b6", "#a78bfa"]

        for i, row in res_df.iterrows():
            fig8.add_trace(go.Bar(
                name=row["Scenario"], x=[row["Scenario"]],
                y=[row["Optimal Tilt (Eff.)"]],
                marker_color=colors[i % len(colors)],
                showlegend=False,
            ), row=1, col=1)
            fig8.add_trace(go.Bar(
                name=row["Scenario"], x=[row["Scenario"]],
                y=[row["Pred. Efficiency"]],
                marker_color=colors[i % len(colors)],
                showlegend=False,
            ), row=1, col=2)

        fig8.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            barmode="group",
            showlegend=False,
        )
        fig8.update_annotations(font=dict(color="#94a3b8", size=12))
        st.plotly_chart(fig8, use_container_width=True)

        # Recommendations per scenario
        st.markdown('<p class="section-header">Per-Scenario Recommendations</p>',
                    unsafe_allow_html=True)
        for i, (sc, row) in enumerate(zip(_sc_inputs, results)):
            with st.expander(f"📌 {row['Scenario']}"):
                recs = get_recommendations(sc["radiation"], sc["loss"],
                                           row["Optimal Tilt (Eff.)"])
                for kind, text in recs:
                    css = "rec-card" if kind == "ok" else "rec-card warn"
                    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="text-align:center; color: #334155; font-size: 0.75rem;
            font-family: 'Space Mono', monospace; margin-top: 48px; padding: 16px 0;
            border-top: 1px solid rgba(255,255,255,0.05);">
    ☀ Solar Optimizer · Powered by NREL PVWatts API + Random Forest
</div>
""", unsafe_allow_html=True)
