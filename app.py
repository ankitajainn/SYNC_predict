"""SYNC Predict - Streamlit prototype. Run with:  streamlit run app.py"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sync_core as sc

st.set_page_config(page_title="SYNC Predict", page_icon="📡", layout="wide")
DATA_PATH = Path(__file__).parent / "data" / "Sync_V1.xlsx"


def clock(t: int) -> str:
    m = 17 * 60 + int(t)
    return f"{m // 60:02d}:{m % 60:02d}"


@st.cache_resource(show_spinner="Calibrating the digital twin and training the models (about 10-30 s)...")
def build(raw: bytes, n_scen: int, seed: int):
    readings, sites, upi = sc.load_data(io.BytesIO(raw))
    cal = sc.calibrate(readings, upi)
    models = sc.train(cal, sites, n_scen=n_scen, seed=seed)
    return readings, sites, upi, cal, models


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Scenario")
fans = st.sidebar.slider("Crowd size (fans)", 50_000, 150_000, 100_000, 5_000)
cap_mult = st.sidebar.slider("B3 capacity multiplier (1.0 = existing network)", 1.0, 2.0, 1.0, 0.1)
gamma = st.sidebar.slider("Ramp shape (>1 = sharper innings-break spike)", 0.6, 2.0, 1.0, 0.1)
alert_thr = st.sidebar.slider("Alert threshold: forecast B3 load (%)", 60, 90, 75, 1)
st.sidebar.header("Twin and model")
n_scen = st.sidebar.slider("Twin scenarios for training", 100, 400, 200, 50)
seed = st.sidebar.number_input("Random seed", 0, 9999, 7)
replay_seed = st.sidebar.number_input("Replay noise seed", 0, 9999, 1)

upload = st.sidebar.file_uploader("Use a different Sync_V1.xlsx (optional)", type=["xlsx"])
if upload is not None:
    raw = upload.getvalue()
elif DATA_PATH.exists():
    raw = DATA_PATH.read_bytes()
else:
    st.title("📡 SYNC Predict")
    st.info("Upload Sync_V1.xlsx in the sidebar, or add it to the repo as data/Sync_V1.xlsx.")
    st.stop()

try:
    readings, sites, upi, cal, models = build(raw, int(n_scen), int(seed))
except ValueError as e:
    st.error(f"Could not read the workbook: {e}")
    st.stop()

st.title("📡 SYNC Predict")
st.caption("Event-aware congestion forecasting for the stadium cells. Trained on a digital twin calibrated to the "
           "32 readings in Sync_V1.xlsx - a prototype to be validated in the rehearsal, not a production system.")
tab1, tab2, tab3, tab4 = st.tabs(["1 · What the data says", "2 · Twin and model", "3 · Live replay", "4 · Scorecard"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed B3 ceiling", f"{cal.cap:.1f}%")
    c2.metric(f"Demand growth at {cal.ref_site}, 17:00 to 21:00", f"{cal.growth_val[1]:.2f}x")
    c3.metric("5G success at 98%+ load", f"{sc.success_pred(cal, [98.3])[0]:.0f}%")

    st.subheader("5G connection success vs 4G anchor (B3) load")
    st.scatter_chart(readings.rename(columns={"m_load": "B3 load % (last match)", "succ": "5G connection success %"}),
                     x="B3 load % (last match)", y="5G connection success %")
    st.caption("Nothing was observed between 81% and 97% load. The twin interpolates that gap (an assumption "
               "to be tested in the rehearsal).")

    st.subheader("The 98% is a ceiling, not demand")
    st.write(f"{cal.ref_site} is the only site that never saturates. If every site grew like it, offered load "
             "(% of B3 capacity) would be:")
    st.dataframe(sc.latent_demand_table(cal, readings))
    st.caption(f"Assumes the other crowds grow like {cal.ref_site}. Values above 100% are demand the cell could not serve.")

    st.subheader("Network averages by match phase")
    summary = readings.groupby("time").agg(b3_load=("m_load", "mean"), success_5g=("succ", "mean"),
                                           speed_4g=("m_4g", "mean"), speed_5g=("m_5g", "mean"),
                                           normal_5g=("n_5g", "mean")).reindex(sc.PHASES)
    summary["5g_speed_retained_%"] = summary["speed_5g"] / summary["normal_5g"] * 100
    st.dataframe(summary.drop(columns="normal_5g").round(1))

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("How it works")
    st.markdown(
        "1. **Uncensor demand** using the unsaturated reference site's growth curve.\n"
        "2. **Digital twin** simulates latent demand per site every 5 min; observed load = min(demand, ceiling).\n"
        "3. **Gradient boosting** learns (from many twin matches with random crowd, capacity and ramp shape) to "
        "forecast latent B3 load 15 min ahead and the probability of crossing 80%.\n"
        "4. **Decision engine** turns forecasts into actions, with the Pay slice and 112 flows never automated."
    )
    st.subheader("Twin vs real last-match readings (65,000 fans)")
    st.dataframe(sc.calibration_check(cal, sites, readings))
    st.caption("The twin was calibrated on these same readings, so this is a sanity check, not validation.")
    lo, hi = models.pressure_range
    st.write(f"Training range of crowd/capacity pressure: {lo:.2f}x to {hi:.2f}x. "
             "Outside it the app falls back to a static rule.")

# ------------------------------------------------------------------ tab 3
with tab3:
    rng = np.random.default_rng(int(replay_seed))
    sim = sc.add_features(sc.simulate(cal, sites, fans, cap_mult, gamma, rng), sites)
    fc = sc.forecast(models, cal, sim)

    t_now = st.slider("Match clock (minutes after 17:00)", 0, 360, 180, 5)
    st.write(f"**Clock: {clock(t_now)}**  ·  innings break at 21:00, final over 22:45, egress 23:15")

    snap = fc[fc["t"] == t_now].copy()
    pressure = float(snap["pressure"].iloc[0])
    in_range = models.pressure_range[0] <= pressure <= models.pressure_range[1]
    if not in_range:
        snap["alert"] = [sc.static_alert(o, b, e) for o, b, e in zip(snap["obs"], snap["to_break"], snap["to_end"])]
        st.warning("Scenario outside the training range: using the static fallback rule instead of the model.")
    else:
        snap["alert"] = snap["forecast"] >= alert_thr

    show = snap[["site", "type", "obs", "forecast", "p_breach", "success_pred", "alert"]].rename(columns={
        "obs": "B3 load now %", "forecast": f"Forecast +{sc.HORIZON} min %", "p_breach": "P(cross 80%)",
        "success_pred": "5G success if unmitigated %", "alert": "ALERT"})
    st.dataframe(show.round(2).set_index("site"))
    mean_load = float(snap["obs"].mean())
    st.caption(f"Network mean B3 load {mean_load:.0f}% → UPI failure lookup ≈ {sc.upi_pred(cal, mean_load):.1f}% "
               "(interpolated from 4 observed points).")

    st.subheader("Recommended actions (all need human approval in this prototype)")
    actions = sc.decide(snap.assign(forecast=snap["forecast"] if in_range else snap["obs"]), t_now, alert_thr)
    if actions.empty:
        st.success("No alerts at this clock time.")
    else:
        st.dataframe(actions.set_index("site"))
    st.caption("Never automated: " + "; ".join(sc.PROTECTED_FLOWS) + ". Slot switching is capped at one third of sites.")

    st.subheader("Forecast vs reality for one site")
    site_pick = st.selectbox("Site", list(sites["site"]), index=0)
    g = fc[fc["site"] == site_pick].sort_values("t")
    chart = pd.DataFrame({"Demand (latent) %": g.set_index("t")["latent"].clip(upper=300),
                          "Observed B3 load %": g.set_index("t")["obs"]})
    past = g[g["t"] <= t_now]
    fcs = pd.Series(past["forecast"].clip(upper=300).values, index=past["t"].values + sc.HORIZON)
    chart["Forecast made 15 min earlier %"] = fcs
    chart.index = [clock(t) for t in chart.index]
    st.line_chart(chart)
    st.caption("The observed line flat-tops at the ceiling; the forecast keeps rising because it learned the demand shape.")

# ------------------------------------------------------------------ tab 4
with tab4:
    res = sc.evaluate(models.oof, alert_thr)
    st.subheader("Twin-based scorecard (held-out simulated matches)")
    st.warning("These numbers measure how well the model recovers the twin it was trained on. They are an upper bound, "
               "not proof of real-world accuracy. The rehearsal is the real test.")
    a, b, c = st.columns(3)
    a.metric("Median alert lead time", f"{res['median_lead_min']:.0f} min", help="Target: at least 10 min before B3 reaches 80%")
    b.metric("Alerts with ≥ 10 min lead", f"{res['lead_ge10'] * 100:.0f}%")
    c.metric("Forecast error (MAPE, 15 min)", f"{res['mape_pct']:.1f}%", help="Target: under 10%")
    d, e, f = st.columns(3)
    d.metric("Breaches missed", f"{res['missed'] * 100:.0f}%")
    e.metric("False alerts (sites that never breach)", f"{res['false_alert'] * 100:.0f}%")
    f.metric("Mean absolute error", f"{res['mae_pts']:.1f} pts")
    st.markdown("**5G connection success ≥ 98%** can only be measured with real or rehearsal traffic after the actions are applied.")
