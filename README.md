# SYNC Predict

Event-aware congestion forecasting prototype for the Airtel SYNC "Stadium Surge" case study.
It calibrates a digital twin to `Sync_V1.xlsx`, trains gradient-boosting models on twin data and
turns 15-minute-ahead forecasts into guard-railed network actions.

## Files
| File | Purpose |
| --- | --- |
| `sync_core.py` | Data loading, demand uncensoring, digital twin, ML models, evaluation, decision engine |
| `app.py` | Streamlit interface (4 tabs) |
| `requirements.txt` | Dependencies |
| `data/Sync_V1.xlsx` | Your dataset (add it yourself; or upload it in the sidebar) |

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. Training takes roughly 10-30 seconds on first load, then is cached.

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repository and push this folder so `app.py` and `requirements.txt` are in the repo root:
   ```bash
   git init && git add . && git commit -m "SYNC Predict"
   git branch -M main
   git remote add origin https://github.com/<your-username>/sync-predict.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose "I already have an app", then select the repository, branch `main`
   and main file path `app.py`.
4. Open **Advanced settings** and pick the same Python version you tested with (3.11 or 3.12 is safe).
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes. Your app gets a
   public `*.streamlit.app` URL you can put in the case-study deck.
6. To update, just `git push`; the app redeploys automatically.

### Notes
- **Data privacy:** a public repo makes `data/Sync_V1.xlsx` public. If the case-study data is
  confidential, add `data/` to `.gitignore` and upload the file through the sidebar instead, or use a
  private repo (you must grant Community Cloud access to private repositories when signing in).
- **Idle apps sleep** on the free tier; the first visitor after a quiet period waits for a restart.
- **Memory:** the default of 200 twin scenarios is light. If the app restarts under load, lower it in the sidebar.

## Plugging in real network data
`add_features()` expects a long dataframe with columns `scenario, site, t, latent, obs, fans, cap_mult`
(`t` = minutes after 17:00 in 5-minute steps; `latent` can be left as `obs` for live data). Replace the
`simulate()` call in the **Live replay** tab with your counters feed (PRB load per sector), then call
`forecast()` and `decide()` unchanged. Add RACH attempts and EN-DC success as extra columns in `FEATURES`
once you have live data for them, and retrain on the twin plus rehearsal data.

## Limitations (say these out loud in the pitch)
- Sync_V1 has 32 readings, no PRB or RACH counters and nothing between snapshots, so the demand ramp
  between 17:00 and 21:00 and the 81-97% load region are twin assumptions.
- The scorecard measures recovery of the twin, not real-world accuracy. Validate in the Day 46-55 rehearsal.
- Actions are recommendations that need human approval; Pay slice and 112 flows are never touched.
