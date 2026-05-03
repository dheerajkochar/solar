# ============================================================
# SOLAR OPTIMIZATION DASHBOARD — Streamlit App
# ============================================================
# Run with:  streamlit run solar_dashboard.py
# Install:   pip install streamlit pandas numpy scikit-learn plotly
# ============================================================

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1e35 0%, #0a1628 100%);
    border-right: 1px solid rgba(251, 191, 36, 0.2);
}
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
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
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
.rec-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(15,30,53,0.8) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.92rem;
    color: #d1fae5;
}
.rec-card.warn {
    background: linear-gradient(135deg, rgba(251,191,36,0.08) 0%, rgba(15,30,53,0.8) 100%);
    border-color: rgba(251, 191, 36, 0.3);
    color: #fef3c7;
}
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

# Single source of truth for dataset filename
DATASET_PATH = "solar_data.csv"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,18,40,0.6)",
    font=dict(family="Sora, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=50, b=40),
)
AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# month excluded — radiation alone encodes seasonality.
# Same radiation → same prediction → physically honest.
FEATURES = ["tilt_angle", "system_loss", "solar_radiation_kwh_m2_day"]

# ============================================================
# DATA & MODEL FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized_power. Drop rows with zero capacity or radiation."""
    df = df.copy()
    df = df[(df["capacity_kw"] > 0) & (df["solar_radiation_kwh_m2_day"] > 0)]
    df["normalized_power"] = df["power_output_w"] / df["capacity_kw"]
    return df.dropna(subset=["normalized_power"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_power_model(df_hash: str, _df: pd.DataFrame):
    """
    Train ONE Random Forest on normalized_power (W/kW).
    Efficiency is derived from predictions — not a separate model —
    so it can never have a negative R².
    """
    df = engineer_features(_df)

    if df.empty:
        st.error("Dataset empty after cleaning. Check your CSV.")
        st.stop()

    X = df[FEATURES]
    y = df["normalized_power"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    rmse   = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2     = float(r2_score(y_te, y_pred))
    return model, rmse, r2


def find_optimal_tilt(model, radiation: float, loss: float,
                      tilt_range: range = range(10, 46)):
    """
    Sweep tilt angles, predict normalized_power for each,
    derive efficiency as power / radiation.
    Returns (best_tilt, best_power, tilts, powers, efficiencies).
    """
    tilt_list = list(tilt_range)

    input_df = pd.DataFrame({
        "tilt_angle":                 tilt_list,
        "system_loss":                [loss]      * len(tilt_list),
        "solar_radiation_kwh_m2_day": [radiation] * len(tilt_list),
    })

    powers       = model.predict(input_df[FEATURES])
    # Efficiency derived from power — no separate broken model
    efficiencies = powers / radiation

    best_idx   = int(np.argmax(powers))
    best_tilt  = tilt_list[best_idx]
    best_power = float(powers[best_idx])

    return best_tilt, best_power, tilt_list, powers.tolist(), efficiencies.tolist()


def get_recommendations(radiation: float, loss: float, optimal_tilt: int) -> list:
    recs = []
    if loss > 12:
        recs.append(("warn", "⚠ System loss > 12% — inspect wiring, connectors, and inverter."))
    else:
        recs.append(("ok",   "✓ System losses are within acceptable range."))
    if radiation < 5.0:
        recs.append(("warn", "⚠ Low irradiance — consider seasonal shading analysis."))
    else:
        recs.append(("ok",   f"✓ Solar irradiance of {radiation:.1f} kWh/m²/day is healthy."))
    if optimal_tilt > 30:
        recs.append(("warn", f"↑ Increase tilt to {optimal_tilt}° for better annual yield."))
    else:
        recs.append(("ok",   f"✓ Tilt of {optimal_tilt}° is near-optimal for this location."))
    return recs


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="hero-badge">☀ Solar AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Configuration")

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


# ============================================================
# DATA LOADING
# ============================================================

df_raw = load_dataset(DATASET_PATH)

# ============================================================
# HERO HEADER
# ============================================================

st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div class="hero-badge">ML-Powered · Random Forest</div>
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
        f"No dataset found at **`{DATASET_PATH}`**. "
        "Place your `solar_data.csv` in the same folder as this script.",
        icon="📡",
    )
    st.stop()

# ============================================================
# VALIDATE COLUMNS
# ============================================================

REQUIRED_COLS = {"month", "tilt_angle", "system_loss",
                 "capacity_kw", "power_output_w", "solar_radiation_kwh_m2_day"}
missing = REQUIRED_COLS - set(df_raw.columns)
if missing:
    st.error(f"CSV missing columns: **{', '.join(sorted(missing))}**\n\nFound: {list(df_raw.columns)}")
    st.stop()

# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner("Training model …"):
    df_hash = (
        str(df_raw.shape)
        + str(df_raw.columns.tolist())
        + str(df_raw.iloc[0].tolist())
        + str(df_raw.iloc[-1].tolist())
    )
    model_power, rmse_p, r2_p = train_power_model(df_hash, df_raw)

df = engineer_features(df_raw)

# ============================================================
# COMPUTE OPTIMAL TILT
# ============================================================

best_tilt, best_power, tilts, powers, efficiencies = find_optimal_tilt(
    model_power, radiation, loss
)
best_efficiency = best_power / radiation

# ============================================================
# TOP METRICS
# ============================================================

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Dataset Rows",        f"{len(df_raw):,}")
m2.metric("Power Model R²",      f"{r2_p:.3f}")
m3.metric("Power Model RMSE",    f"{rmse_p:.2f}")
m4.metric("Optimal Tilt",        f"{best_tilt}°")
m5.metric("Peak Efficiency",     f"{best_efficiency:.2f} W/kWh·m⁻²")

# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs([
    "🎯  Tilt Optimizer",
    "📊  Data Explorer",
])


# ─────────────────────────────────────────────────────────────
# TAB 1 — TILT OPTIMIZER
# ─────────────────────────────────────────────────────────────

with tab1:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<p class="section-header">Tilt Angle vs. Predicted Output</p>',
                    unsafe_allow_html=True)

        fig = go.Figure()

        # Power trace (primary y-axis)
        fig.add_trace(go.Scatter(
            x=tilts, y=powers,
            mode="lines+markers",
            name="Normalised Power (W/kW)",
            line=dict(color="#fbbf24", width=2.5),
            marker=dict(size=5),
            yaxis="y1",
        ))

        # Efficiency trace (secondary y-axis) — derived, not from broken model
        fig.add_trace(go.Scatter(
            x=tilts, y=efficiencies,
            mode="lines+markers",
            name="Efficiency (W / kWh·m⁻²·day)",
            line=dict(color="#34d399", width=2.5, dash="dot"),
            marker=dict(size=5),
            yaxis="y2",
        ))

        fig.add_vline(
            x=best_tilt, line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Optimal {best_tilt}°",
            annotation_font_color="#fbbf24",
        )

        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis =dict(title="Tilt Angle (°)",          **AXIS_STYLE),
            yaxis =dict(title="Normalised Power (W/kW)",  color="#fbbf24", **AXIS_STYLE),
            yaxis2=dict(title="Efficiency",               color="#34d399",
                        overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-header">System Recommendations</p>',
                    unsafe_allow_html=True)

        for kind, text in get_recommendations(radiation, loss, best_tilt):
            css = "rec-card" if kind == "ok" else "rec-card warn"
            st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header" style="margin-top:24px;">Optimal Config</p>',
                    unsafe_allow_html=True)

        cfg = pd.DataFrame({
            "Parameter": ["Optimal Tilt", "System Loss", "Radiation", "Capacity"],
            "Value":     [f"{best_tilt}°", f"{loss}%",
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

        df_cap   = df[df["capacity_kw"] == capacity].copy()
        df_pivot = (
            df_cap.groupby(["month", "tilt_angle"])["normalized_power"]
            .mean().reset_index()
        )

        fig2 = px.line(
            df_pivot, x="month", y="normalized_power",
            color="tilt_angle",
            labels={"month": "Month", "normalized_power": "W/kW", "tilt_angle": "Tilt (°)"},
            color_discrete_sequence=px.colors.sequential.YlOrBr,
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=340)
        fig2.update_xaxes(tickvals=list(range(1, 13)), ticktext=MONTH_NAMES, **AXIS_STYLE)
        fig2.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Power vs Solar Radiation (scatter)</p>',
                    unsafe_allow_html=True)

        sample = df.sample(min(2000, len(df)), random_state=1)
        fig3 = px.scatter(
            sample,
            x="solar_radiation_kwh_m2_day", y="normalized_power",
            color="tilt_angle",
            labels={
                "solar_radiation_kwh_m2_day": "Solar Radiation (kWh/m²/day)",
                "normalized_power": "Normalised Power (W/kW)",
                "tilt_angle": "Tilt (°)",
            },
            color_continuous_scale="YlOrBr",
        )
        fig3.update_layout(**PLOTLY_LAYOUT, height=340,
                           coloraxis_colorbar=dict(title="Tilt (°)"))
        fig3.update_traces(marker=dict(opacity=0.6, size=4))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section-header">Monthly Average Solar Radiation Heatmap</p>',
                unsafe_allow_html=True)

    heat = (
        df.groupby(["month", "tilt_angle"])["solar_radiation_kwh_m2_day"]
        .mean().reset_index()
    )
    heat_pivot = heat.pivot(
        index="tilt_angle", columns="month",
        values="solar_radiation_kwh_m2_day"
    )
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


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="text-align:center; color: #334155; font-size: 0.75rem;
            font-family: 'Space Mono', monospace; margin-top: 48px; padding: 16px 0;
            border-top: 1px solid rgba(255,255,255,0.05);">
    ☀ Solar Optimizer · Random Forest · solar_data.csv
</div>
""", unsafe_allow_html=True)
