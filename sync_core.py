"""
SYNC Predict - core logic (no Streamlit imports, so it can be tested on its own).

Pipeline
  1. load_data()      read Sync_V1.xlsx (32 site-phase readings, 8 sites, 4 UPI points)
  2. calibrate()      uncensor demand (the ~98% B3 load is a capacity ceiling) and fit
                      monotone curves for 5G success and 4G speed
  3. simulate()       digital twin: latent demand per site every 5 min, observed = min(demand, cap)
  4. train()          gradient boosting on twin data: load 15 min ahead + breach probability
  5. evaluate()       out-of-fold accuracy and alert lead time (twin-based, NOT real-world proof)
  6. decide()         turn forecasts into guard-railed actions
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

# ----------------------------------------------------------------------------- constants
PHASES = ["17:00", "21:00", "22:45", "23:15"]
PHASE_MIN = {"17:00": 0, "21:00": 240, "22:45": 345, "23:15": 375}   # minutes after 17:00
BREAK_MIN, FINAL_MIN, END_MIN = 240, 345, 375
STEP, HORIZON = 5, 15            # minutes
REF_FANS = 65_000                # crowd size of the last match in Sync_V1
BREACH = 80.0                    # B3 load % above which 5G success starts to slip
LATENT_CLIP = 400.0              # offered load is capped at 400% of capacity in the twin
FEATURES = ["obs", "obs_prev15", "slope", "load0", "t_min", "to_break", "to_end",
            "fans_k", "cap_mult", "pressure", "is_small", "backhaul_gbps", "dist_m", "has_5g"]
PROTECTED_FLOWS = ["112 emergency (5QI 1, ARP 1)", "UPI / Pay slice (5QI 5)"]


# ----------------------------------------------------------------------------- 1. data
def _clock(v) -> str:
    return v.strftime("%H:%M") if hasattr(v, "strftime") else str(v).strip()[:5]


def load_data(src):
    """src: path or file-like object holding Sync_V1.xlsx. Returns (readings, sites, upi)."""
    raw = pd.read_excel(src, header=None)
    if str(raw.iloc[1, 0]).strip() != "Site" or str(raw.iloc[1, 11]).strip() != "Site":
        raise ValueError("Unexpected layout: expected the Sync_V1.xlsx template (headers on row 2).")

    rd = raw.iloc[2:34, 0:10].copy()
    rd.columns = ["site", "phase", "time", "n_load", "m_load", "n_4g", "m_4g", "n_5g", "m_5g", "succ"]
    rd["site"] = rd["site"].astype(str).str.strip()
    rd["time"] = rd["time"].map(_clock)
    for c in ["n_load", "m_load", "n_4g", "m_4g", "n_5g", "m_5g", "succ"]:
        rd[c] = pd.to_numeric(rd[c], errors="coerce")
    rd = rd.reset_index(drop=True)

    sites = raw.iloc[2:10, 11:17].copy()
    sites.columns = ["site", "type", "dist_m", "backhaul", "backhaul_gbps", "has_5g"]
    sites["site"] = sites["site"].astype(str).str.strip()
    sites["dist_m"] = pd.to_numeric(sites["dist_m"])
    sites["backhaul_gbps"] = pd.to_numeric(sites["backhaul_gbps"])
    sites["has_5g"] = sites["has_5g"].astype(str).str.lower().eq("yes").astype(int)
    sites["is_small"] = sites["type"].astype(str).str.contains("Small").astype(int)
    sites = sites.reset_index(drop=True)

    upi = raw.iloc[13:17, 11:14].copy()
    upi.columns = ["time", "upi_match", "upi_normal"]
    upi["time"] = upi["time"].map(_clock)
    upi[["upi_match", "upi_normal"]] = upi[["upi_match", "upi_normal"]].apply(pd.to_numeric)
    return rd, sites, upi.reset_index(drop=True)


# ----------------------------------------------------------------------------- 2. calibration
@dataclass
class Calibration:
    cap: float                 # observed B3 ceiling (%)
    ref_site: str              # the one site that never saturates
    growth_min: np.ndarray     # anchor times (minutes after 17:00)
    growth_val: np.ndarray     # demand multiplier vs 17:00 at those times
    base17: dict               # site -> observed B3 load at 17:00
    succ_model: IsotonicRegression
    speed4g_model: IsotonicRegression
    upi_x: np.ndarray
    upi_y: np.ndarray


def calibrate(readings: pd.DataFrame, upi: pd.DataFrame, ref_site: str | None = None) -> Calibration:
    ld = readings.pivot(index="site", columns="time", values="m_load")[PHASES]
    cap = float(readings.loc[readings["m_load"] >= 95, "m_load"].median())

    if ref_site is None:                          # least-loaded site at innings break
        ref_site = ld["21:00"].idxmin()
    if ld.loc[ref_site].max() > cap - 5:
        raise ValueError(f"Reference site {ref_site} is saturated; demand cannot be uncensored from it.")
    growth = (ld.loc[ref_site] / ld.loc[ref_site, "17:00"]).values

    succ = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(readings["m_load"], readings["succ"])
    ok = readings.dropna(subset=["m_4g"])
    spd = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(ok["m_load"], ok["m_4g"])

    mean_load = readings.groupby("time")["m_load"].mean().reindex(PHASES).values
    ux = np.append(readings["n_load"].mean(), mean_load)
    uy = np.append(upi["upi_normal"].mean(), upi.set_index("time").loc[PHASES, "upi_match"].values)
    order = np.argsort(ux)
    return Calibration(cap, ref_site, np.array([PHASE_MIN[p] for p in PHASES], float), growth,
                       ld["17:00"].to_dict(), succ, spd, ux[order], np.maximum.accumulate(uy[order]))


def growth_curve(cal: Calibration, t, gamma: float = 1.0):
    """Demand multiplier vs 17:00. gamma>1 delays the ramp (sharper innings-break spike)."""
    t = np.asarray(t, float)
    g21 = cal.growth_val[1]
    ramp = 1 + (g21 - 1) * np.clip(t / BREAK_MIN, 0, 1) ** gamma
    rest = np.interp(t, cal.growth_min[1:], cal.growth_val[1:])
    return np.where(t <= BREAK_MIN, ramp, rest)


def latent_demand_table(cal: Calibration, readings: pd.DataFrame) -> pd.DataFrame:
    """Implied offered load (% of B3 capacity) if every site grew like the reference site."""
    out = {}
    for ph, g in zip(PHASES, cal.growth_val):
        out[ph] = {s: round(b * g, 0) for s, b in cal.base17.items()}
    return pd.DataFrame(out)


def success_pred(cal, load):
    return cal.succ_model.predict(np.minimum(np.atleast_1d(np.asarray(load, float)), cal.cap))


def upi_pred(cal, mean_load):
    return float(np.interp(mean_load, cal.upi_x, cal.upi_y))


# ----------------------------------------------------------------------------- 3. digital twin
def simulate(cal, sites, fans=REF_FANS, cap_mult=1.0, gamma=1.0, rng=None, noise=True, scenario=0):
    """One match: latent demand and observed (capped) B3 load per site every STEP minutes."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(0, END_MIN + 1, STEP)
    g = growth_curve(cal, t, gamma)
    frames = []
    for s in sites["site"]:
        site_f = rng.lognormal(0, 0.06) if noise else 1.0
        lat = cal.base17[s] * site_f * (fans / REF_FANS) * g / cap_mult
        if noise:
            lat = lat * rng.lognormal(0, 0.02, size=len(t))
        frames.append(pd.DataFrame({"scenario": scenario, "site": s, "t": t, "latent": lat,
                                    "obs": np.minimum(lat, cal.cap),
                                    "fans": fans, "cap_mult": cap_mult, "gamma": gamma}))
    return pd.concat(frames, ignore_index=True)


def add_features(df: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    lag = HORIZON // STEP
    df = df.sort_values(["scenario", "site", "t"]).reset_index(drop=True)
    g = df.groupby(["scenario", "site"], sort=False)
    df["obs_prev15"] = g["obs"].shift(lag).fillna(g["obs"].transform("first"))
    df["slope"] = (df["obs"] - df["obs_prev15"]) / HORIZON
    df["load0"] = g["obs"].transform("first")
    df["y"] = g["latent"].shift(-lag)                       # truth 15 min ahead (NaN at the end)
    df["t_min"], df["to_break"], df["to_end"] = df["t"], BREAK_MIN - df["t"], END_MIN - df["t"]
    df["fans_k"] = df["fans"] / 1000
    df["pressure"] = df["fans"] / REF_FANS / df["cap_mult"]
    return df.merge(sites[["site", "type", "is_small", "backhaul_gbps", "dist_m", "has_5g"]], on="site", how="left")


# ----------------------------------------------------------------------------- 4. training
@dataclass
class Models:
    reg: HistGradientBoostingRegressor
    clf: HistGradientBoostingClassifier
    oof: pd.DataFrame
    pressure_range: tuple


def train(cal, sites, n_scen=200, seed=7, n_splits=3) -> Models:
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_scen):                  # randomise crowd, capacity and ramp shape
        d = simulate(cal, sites, fans=rng.uniform(50_000, 150_000), cap_mult=rng.uniform(1.0, 1.8),
                     gamma=rng.uniform(0.6, 2.0), rng=rng, scenario=i)
        parts.append(d)
    data = add_features(pd.concat(parts, ignore_index=True), sites).dropna(subset=["y"]).reset_index(drop=True)

    X, y = data[FEATURES], data["y"].clip(upper=LATENT_CLIP)
    lab = (data["y"] >= BREACH).astype(int)
    mk_r = lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=seed)
    mk_c = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)

    pred, prob = np.zeros(len(data)), np.zeros(len(data))
    for tr, te in GroupKFold(n_splits).split(X, y, data["scenario"]):   # whole matches held out
        pred[te] = mk_r().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        prob[te] = mk_c().fit(X.iloc[tr], lab.iloc[tr]).predict_proba(X.iloc[te])[:, 1]

    oof = data[["scenario", "site", "t", "latent", "y"]].assign(pred=pred, p_breach=prob)
    return Models(mk_r().fit(X, y), mk_c().fit(X, lab), oof,
                  (float(data["pressure"].min()), float(data["pressure"].max())))


# ----------------------------------------------------------------------------- 5. evaluation
def evaluate(oof: pd.DataFrame, alert_thr: float = 75.0) -> dict:
    y, p = oof["y"].clip(upper=LATENT_CLIP), oof["pred"]
    res = {"mae_pts": float((p - y).abs().mean()),
           "mape_pct": float(((p - y).abs() / y.clip(lower=20)).mean() * 100)}
    actual, flagged = oof["y"] >= BREACH, p >= alert_thr
    res["precision"] = float((actual & flagged).sum() / max(flagged.sum(), 1))
    res["recall"] = float((actual & flagged).sum() / max(actual.sum(), 1))

    leads, missed, false_alert, n_site, n_clean = [], 0, 0, 0, 0
    for _, g in oof.groupby(["scenario", "site"]):
        g = g.sort_values("t")
        breach_t = g.loc[g["latent"] >= BREACH, "t"].min()
        alert_t = g.loc[g["pred"] >= alert_thr, "t"].min()
        if pd.isna(breach_t):
            n_clean += 1
            false_alert += int(not pd.isna(alert_t))
            continue
        if breach_t == 0:                    # already in breach at 17:00 - nothing to forecast
            continue
        n_site += 1
        if pd.isna(alert_t) or alert_t > breach_t:
            missed += 1
        else:
            leads.append(breach_t - alert_t)
    leads = np.array(leads) if leads else np.array([0.0])
    res.update(median_lead_min=float(np.median(leads)), lead_ge10=float((leads >= 10).sum() / max(n_site, 1)),
               missed=float(missed / max(n_site, 1)), false_alert=float(false_alert / max(n_clean, 1)))
    return res


def calibration_check(cal, sites, readings) -> pd.DataFrame:
    """Twin at 65,000 fans vs the real last-match readings (the twin was calibrated on these)."""
    tw = simulate(cal, sites, noise=False)
    rows = []
    for _, r in readings.iterrows():
        sim = tw[(tw["site"] == r["site"]) & (tw["t"] == PHASE_MIN[r["time"]])].iloc[0]
        rows.append({"site": r["site"], "time": r["time"], "real_load": r["m_load"], "twin_load": round(sim["obs"], 1),
                     "real_5g_success": r["succ"], "twin_5g_success": round(float(success_pred(cal, sim["obs"])[0]), 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 6. forecast + decide
def forecast(models: Models, cal: Calibration, feat: pd.DataFrame) -> pd.DataFrame:
    X = feat[FEATURES]
    out = feat.copy()
    out["forecast"] = models.reg.predict(X)
    out["p_breach"] = models.clf.predict_proba(X)[:, 1]
    out["success_pred"] = success_pred(cal, out["forecast"])
    return out


def static_alert(obs, to_break, to_end) -> bool:
    """Fallback rule when inputs fall outside the range the model was trained on."""
    near_event = (0 <= to_break <= 30) or (0 <= to_end <= 30) or to_break <= 0
    return bool(obs >= 60 and near_event)


def decide(snap: pd.DataFrame, t_min: int, alert_thr: float = 75.0, max_slot_share: float = 1 / 3) -> pd.DataFrame:
    """snap: one row per site with site, forecast, has_5g, type. Returns recommended actions."""
    acts = []
    hot = snap[snap["forecast"] >= alert_thr].sort_values("forecast", ascending=False)
    spare = snap[(snap["forecast"] < 60) & (snap["type"] == "Macro")].sort_values("forecast")
    slot_budget = int(len(snap) * max_slot_share)
    near_break = abs(t_min - BREAK_MIN) <= 30

    for _, r in hot.iterrows():
        if r["has_5g"]:
            acts.append((r["site"], "SA fast-track + raise n78 reselection priority", "forecast breaches the load knee"))
            acts.append((r["site"], "Reserve PRBs for SgNB (EN-DC) addition", "protects 5G attach"))
            if near_break and slot_budget > 0:
                acts.append((r["site"], "Switch 4:1 -> 3:2 slot pattern", "UL surge window"))
                slot_budget -= 1
        else:
            acts.append((r["site"], "4G mobility offset toward neighbours", "no n78 on this site"))
        if len(spare):
            acts.append((r["site"], f"Offload to {spare.iloc[0]['site']} (forecast {spare.iloc[0]['forecast']:.0f}%)", "spare headroom"))
    if t_min >= FINAL_MIN - 15:
        acts.append(("CoW fleet", "Pre-tilt CoWs to gate plazas; revert 3:2 -> 4:1", "egress surge"))
    df = pd.DataFrame(acts, columns=["site", "action", "reason"])
    df["needs_approval"] = True              # human-in-the-loop until the rehearsal proves the model
    return df
