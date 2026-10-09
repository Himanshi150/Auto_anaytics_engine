# Auto-Analytics Engine

A general-purpose Streamlit dashboard for detecting healthcare performance trends, IQR outliers, Pearson correlations, and human-readable insights from district-level CSV data.

## Features

- Loads the bundled CSV or a compatible uploaded CSV.
- Shows `head()`, `info()`, and missing-value counts.
- Live filters for district, month, and indicator.
- Configurable trend, IQR, and correlation thresholds.
- Data-driven insight rows with IDs, type, indicator, entity, period, values, severity, and explanation.
- Insight CSV download.
- Severity bar chart, per-district indicator line chart, and Pearson correlation heatmap.

## Run on Windows

Install Python 3.10 or newer, then open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## GitHub: Himanshi150

This folder is prepared to be pushed to the GitHub account or organization associated with `Himanshi150`. GitHub authentication and repository ownership cannot be changed from the local project without your credentials.

After creating a repository named `auto-analytics-engine` under that account:

```powershell
git init
git add .
git commit -m "Complete automated insight generation dashboard"
git branch -M main
git remote add origin https://github.com/Himanshi150/auto-analytics-engine.git
git push -u origin main
```

If `Himanshi150` is a collaborator's account, invite it from the repository's GitHub Settings, then push using an account with write access.

## Notes

The provided sample has only 12 observations. Pearson correlations and IQR boundaries are therefore useful screening signals, not proof of causation. The application reports the matrix honestly and keeps thresholds configurable.
