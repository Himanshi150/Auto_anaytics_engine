from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "healthcare_performance.csv"
REQUIRED_COLUMNS = [
    "month",
    "district",
    "anc_coverage",
    "institutional_delivery",
    "immunization",
    "high_risk_cases",
]
INDICATORS = [column for column in REQUIRED_COLUMNS if column not in {"month", "district"}]


def load_data(uploaded_file=None) -> pd.DataFrame:
    source = uploaded_file if uploaded_file is not None else DATA_PATH
    frame = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame[REQUIRED_COLUMNS].copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    for indicator in INDICATORS:
        frame[indicator] = pd.to_numeric(frame[indicator], errors="coerce")
    if frame["month"].isna().any():
        raise ValueError("The month column contains invalid dates.")
    return frame.sort_values(["district", "month"]).reset_index(drop=True)


def missing_report(frame: pd.DataFrame) -> pd.Series:
    return frame.isna().sum()


def detect_trends(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (district, indicator), group in frame.groupby(["district", "indicator"]):
        ordered = group.sort_values("month")
        if len(ordered) < 2:
            continue
        previous = ordered.iloc[-2]
        current = ordered.iloc[-1]
        if previous["value"] == 0:
            continue
        change_pct = (current["value"] - previous["value"]) / abs(previous["value"]) * 100
        if abs(change_pct) >= threshold:
            rows.append(
                {
                    "type": "trend",
                    "indicator": indicator,
                    "entity": district,
                    "period": current["month"].strftime("%Y-%m"),
                    "value": current["value"],
                    "prev_value": previous["value"],
                    "change_pct": change_pct,
                    "severity": severity_for_change(change_pct, threshold),
                }
            )
    return pd.DataFrame(rows)


def detect_outliers(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    rows = []
    for indicator, group in frame.groupby("indicator"):
        q1 = group["value"].quantile(0.25)
        q3 = group["value"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        outliers = group[(group["value"] < lower) | (group["value"] > upper)]
        for _, record in outliers.iterrows():
            distance = min(abs(record["value"] - lower), abs(record["value"] - upper))
            scale = max(abs(iqr), 1e-9)
            rows.append(
                {
                    "type": "outlier",
                    "indicator": indicator,
                    "entity": record["district"],
                    "period": record["month"].strftime("%Y-%m"),
                    "value": record["value"],
                    "prev_value": np.nan,
                    "change_pct": np.nan,
                    "severity": "High" if distance / scale >= 2 else "Medium",
                }
            )
    return pd.DataFrame(rows)


def detect_correlations(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = frame.pivot_table(index=["district", "month"], columns="indicator", values="value")
    matrix = pivot.corr(method="pearson")
    rows = []
    for index, first in enumerate(matrix.columns):
        for second in matrix.columns[index + 1 :]:
            correlation = matrix.loc[first, second]
            if pd.notna(correlation) and abs(correlation) >= threshold:
                rows.append(
                    {
                        "type": "correlation",
                        "indicator": f"{first}:{second}",
                        "entity": "Statewide",
                        "period": "All available months",
                        "value": correlation,
                        "prev_value": np.nan,
                        "change_pct": np.nan,
                        "severity": "High" if abs(correlation) >= 0.85 else "Medium",
                    }
                )
    return matrix, pd.DataFrame(rows)


def severity_for_change(change_pct: float, threshold: float) -> str:
    ratio = abs(change_pct) / max(threshold, 0.01)
    if ratio >= 2:
        return "High"
    if ratio >= 1.5:
        return "Medium"
    return "Low"


def build_explanation(row: pd.Series, trend_threshold: float) -> str:
    if row["type"] == "trend":
        direction = "increased" if row["change_pct"] > 0 else "decreased"
        return (
            f"{row['indicator']} in {row['entity']} {direction} by "
            f"{abs(row['change_pct']):.1f}% from {row['prev_value']:.1f} to {row['value']:.1f}, "
            f"exceeding the {trend_threshold:.0f}% significant-change threshold."
        )
    if row["type"] == "outlier":
        return (
            f"{row['indicator']} for {row['entity']} was an IQR outlier at "
            f"{row['value']:.1f} during {row['period']}, requiring review."
        )
    first, second = row["indicator"].split(":", maxsplit=1)
    relationship = "positive" if row["value"] > 0 else "negative"
    return (
        f"{first} and {second} show a {relationship} Pearson correlation of "
        f"{row['value']:.2f} across the available observations."
    )


def build_insights(frame: pd.DataFrame, trend_threshold: float, outlier_multiplier: float, correlation_threshold: float) -> pd.DataFrame:
    long_frame = frame.melt(
        id_vars=["month", "district"], value_vars=INDICATORS, var_name="indicator", value_name="value"
    ).dropna(subset=["value"])
    trend_rows = detect_trends(long_frame, trend_threshold)
    outlier_rows = detect_outliers(long_frame, outlier_multiplier)
    _, correlation_rows = detect_correlations(long_frame, correlation_threshold)
    insights = pd.concat([trend_rows, outlier_rows, correlation_rows], ignore_index=True)
    if insights.empty:
        return pd.DataFrame(columns=["insight_id", "type", "indicator", "entity", "period", "value", "prev_value", "change_pct", "severity", "explanation"])
    insights["explanation"] = insights.apply(lambda row: build_explanation(row, trend_threshold), axis=1)
    insights.insert(0, "insight_id", [f"INS-{number:04d}" for number in range(1, len(insights) + 1)])
    return insights[["insight_id", "type", "indicator", "entity", "period", "value", "prev_value", "change_pct", "severity", "explanation"]]


def make_download(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


st.set_page_config(page_title="Auto-Analytics Engine", page_icon="AA", layout="wide")
st.title("Auto-Analytics Engine")
st.caption("District healthcare performance monitoring with data-driven insights")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload a compatible CSV", type="csv")
    trend_threshold = st.slider("Significant change threshold (%)", 1, 50, 10)
    outlier_multiplier = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1)
    correlation_threshold = st.slider("Correlation threshold (|r|)", 0.50, 0.99, 0.70, 0.01)

try:
    data = load_data(uploaded_file)
except (OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.expander("Data validation", expanded=False):
    st.write("Preview")
    st.dataframe(data.head(), use_container_width=True, hide_index=True)
    st.write("DataFrame info")
    info_buffer = io.StringIO()
    data.info(buf=info_buffer)
    st.code(info_buffer.getvalue())
    st.write("Missing values per column")
    st.dataframe(missing_report(data).rename("missing_values"), use_container_width=True)

all_months = sorted(data["month"].dt.strftime("%Y-%m").unique())
all_districts = sorted(data["district"].unique())
selected_districts = st.multiselect("District", all_districts, default=all_districts)
selected_months = st.multiselect("Month", all_months, default=all_months)
selected_indicators = st.multiselect("Indicator", INDICATORS, default=INDICATORS)

filtered = data[
    data["district"].isin(selected_districts)
    & data["month"].dt.strftime("%Y-%m").isin(selected_months)
].copy()
long_filtered = filtered.melt(
    id_vars=["month", "district"], value_vars=selected_indicators,
    var_name="indicator", value_name="value",
).dropna(subset=["value"])

insights = build_insights(filtered, trend_threshold, outlier_multiplier, correlation_threshold)
if selected_indicators:
    insights = insights[insights["indicator"].str.split(":").str[0].isin(selected_indicators)]

metric_columns = st.columns(4)
metric_columns[0].metric("Rows loaded", len(data))
metric_columns[1].metric("Insights", len(insights))
metric_columns[2].metric("High severity", int((insights["severity"] == "High").sum()))
metric_columns[3].metric("Districts", data["district"].nunique())

st.subheader("Generated insights")
if insights.empty:
    st.info("No findings match the current filters and thresholds.")
else:
    st.dataframe(insights, use_container_width=True, hide_index=True)
    st.download_button("Download insights CSV", make_download(insights), "insights.csv", "text/csv")

left, right = st.columns(2)
with left:
    st.subheader("Severity counts")
    severity_counts = insights["severity"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("severity").reset_index(name="count")
    st.plotly_chart(px.bar(severity_counts, x="severity", y="count", color="severity", color_discrete_map={"Low": "#4c956c", "Medium": "#f4a261", "High": "#d1495b"}), use_container_width=True)
with right:
    st.subheader("Indicator trends")
    if long_filtered.empty:
        st.info("Select at least one indicator.")
    else:
        chart = px.line(long_filtered, x="month", y="value", color="district", facet_row="indicator", markers=True)
        chart.update_yaxes(matches=None)
        st.plotly_chart(chart, use_container_width=True)

st.subheader("Pearson correlation heatmap")
correlation_matrix, _ = detect_correlations(long_filtered, correlation_threshold)
if correlation_matrix.empty:
    st.info("Not enough numeric observations for a correlation matrix.")
else:
    heatmap = px.imshow(correlation_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(heatmap, use_container_width=True)
    st.download_button("Download correlation matrix CSV", make_download(correlation_matrix.reset_index()), "correlation_matrix.csv", "text/csv")

st.caption("Correlation estimates are fragile with small samples. Interpret them as signals for review, not causal evidence.")
