#!/usr/bin/env python3
"""
Streamflow Transfer Learning Dashboard

Interactive visualization of NeuralHydrology model predictions,
metrics, and training runs for Lamar and Hoh watersheds.
"""

import pickle
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Streamflow Forecast Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"

WATERSHED_COLORS = {
    "lamar": "#1f77b4",
    "hoh": "#2ca02c",
}

# ── Data loading helpers ──────────────────────────────────────────────────────

@st.cache_data
def discover_runs():
    """Find all runs that have test results."""
    runs = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        for metrics_file in sorted(run_dir.glob("test/*/test_metrics.csv")):
            epoch_dir = metrics_file.parent
            epoch = epoch_dir.name  # e.g. model_epoch030
            results_file = epoch_dir / "test_results.p"
            config_file = run_dir / "config.yml"
            if results_file.exists() and config_file.exists():
                df_metrics = pd.read_csv(metrics_file)
                runs.append({
                    "run_name": run_dir.name,
                    "run_dir": run_dir,
                    "epoch": epoch,
                    "results_file": results_file,
                    "config_file": config_file,
                    "basin": df_metrics["basin"].iloc[0],
                    "NSE": df_metrics["NSE"].iloc[0],
                    "KGE": df_metrics["KGE"].iloc[0],
                    "RMSE": df_metrics["RMSE"].iloc[0],
                    "Pearson-r": df_metrics["Pearson-r"].iloc[0],
                })
    return pd.DataFrame(runs)


@st.cache_data
def load_run_data(results_file: str, config_file: str):
    """Load predictions and config for a single run."""
    with open(results_file, "rb") as f:
        results = pickle.load(f)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    basin = list(results.keys())[0]
    xr_data = results[basin]["1D"]["xr"]

    dates = pd.to_datetime(xr_data["date"].values)
    obs = xr_data["QObs_mm_d_obs"].values.squeeze().astype(float)
    pred = xr_data["QObs_mm_d_sim"].values.squeeze().astype(float)

    df = pd.DataFrame({"date": dates, "obs": obs, "pred": pred})
    df = df.dropna()
    return df, config, basin


def compute_metrics(obs, pred):
    nse = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    bias = np.mean(pred - obs)
    r = np.corrcoef(obs, pred)[0, 1]
    return dict(NSE=nse, RMSE=rmse, MAE=mae, Bias=bias, R2=r**2, Pearson_r=r)


def split_periods(df, config):
    """Return train / val / test DataFrames based on config dates."""
    train_end = pd.to_datetime(config["train_end_date"], dayfirst=True)
    val_end = pd.to_datetime(config["validation_end_date"], dayfirst=True)
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test


# ── Plot helpers ─────────────────────────────────────────────────────────────

def fig_timeseries(df, config, title, show_splits=True):
    train_end = pd.to_datetime(config["train_end_date"], dayfirst=True)
    val_end = pd.to_datetime(config["validation_end_date"], dayfirst=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["obs"],
        name="Observed", line=dict(color="black", width=1), opacity=0.75,
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["pred"],
        name="Predicted", line=dict(color="crimson", width=1), opacity=0.85,
    ))

    if show_splits:
        for xval, label, color in [
            (train_end, "Train/Val", "royalblue"),
            (val_end, "Val/Test", "seagreen"),
        ]:
            if df["date"].min() <= xval <= df["date"].max():
                fig.add_vline(x=xval, line_dash="dash", line_color=color,
                              annotation_text=label, annotation_position="top right")

    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Streamflow (mm/day)",
        hovermode="x unified", legend=dict(orientation="h", y=1.05),
        height=380, margin=dict(t=60, b=40),
    )
    return fig


def fig_scatter(obs, pred, r2):
    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    z = np.polyfit(obs, pred, 1)
    p = np.poly1d(z)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs, y=pred, mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.5),
        name="Obs vs Pred",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="black", dash="dash"), name="1:1",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[p(lo), p(hi)], mode="lines",
        line=dict(color="crimson"), name=f"Fit: {z[0]:.2f}x+{z[1]:.2f}",
    ))
    fig.update_layout(
        title=f"Scatter (R² = {r2:.3f})",
        xaxis_title="Observed (mm/day)", yaxis_title="Predicted (mm/day)",
        height=360, margin=dict(t=50, b=40),
    )
    return fig


def fig_residuals(obs, pred, mae):
    residuals = pred - obs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=obs, y=residuals, mode="markers",
        marker=dict(color="coral", size=4, opacity=0.5),
        name="Residual",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    mean_r = residuals.mean()
    std_r = residuals.std()
    fig.add_hline(y=mean_r, line_color="red",
                  annotation_text=f"Mean: {mean_r:.3f}", annotation_position="right")
    fig.add_hline(y=mean_r + std_r, line_dash="dot", line_color="red", opacity=0.6)
    fig.add_hline(y=mean_r - std_r, line_dash="dot", line_color="red", opacity=0.6)
    fig.update_layout(
        title=f"Residuals (MAE = {mae:.3f})",
        xaxis_title="Observed (mm/day)", yaxis_title="Residual (mm/day)",
        height=360, margin=dict(t=50, b=40),
    )
    return fig


def fig_monthly(df):
    df = df.copy()
    df["month"] = df["date"].dt.month
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for m in range(1, 13):
        sub = df[df["month"] == m].dropna()
        if len(sub) >= 10:
            m_obs, m_pred = sub["obs"].values, sub["pred"].values
            nse = 1 - np.sum((m_obs - m_pred) ** 2) / np.sum((m_obs - np.mean(m_obs)) ** 2)
            rmse = np.sqrt(np.mean((m_obs - m_pred) ** 2))
            rows.append({"month": m, "month_name": month_names[m - 1], "NSE": nse, "RMSE": rmse})

    if not rows:
        return None, None

    mdf = pd.DataFrame(rows)

    colors_nse = ["green" if v > 0.7 else "orange" if v > 0.5 else "red" for v in mdf["NSE"]]

    fig1 = go.Figure(go.Bar(
        x=mdf["month_name"], y=mdf["NSE"],
        marker_color=colors_nse, name="NSE",
    ))
    fig1.add_hline(y=0.7, line_dash="dash", line_color="green",
                   annotation_text="Good (0.7)", annotation_position="right")
    fig1.add_hline(y=0.5, line_dash="dash", line_color="orange",
                   annotation_text="Fair (0.5)", annotation_position="right")
    fig1.update_layout(title="Monthly NSE", yaxis_title="NSE",
                       height=300, margin=dict(t=50, b=30))

    fig2 = go.Figure(go.Bar(
        x=mdf["month_name"], y=mdf["RMSE"],
        marker_color="steelblue", name="RMSE",
    ))
    mean_rmse = mdf["RMSE"].mean()
    fig2.add_hline(y=mean_rmse, line_dash="dash", line_color="red",
                   annotation_text=f"Mean: {mean_rmse:.3f}", annotation_position="right")
    fig2.update_layout(title="Monthly RMSE", yaxis_title="RMSE (mm/day)",
                       height=300, margin=dict(t=50, b=30))

    return fig1, fig2


def fig_flow_duration(obs, pred):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs, pred = obs[mask], pred[mask]
    n = len(obs)
    exc = np.arange(1, n + 1) / n * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exc, y=np.sort(obs)[::-1],
        name="Observed", line=dict(color="black", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=exc, y=np.sort(pred)[::-1],
        name="Predicted", line=dict(color="crimson", width=2), opacity=0.85,
    ))
    for p in [5, 25, 50, 75, 95]:
        fig.add_vline(x=p, line_dash="dot", line_color="gray", opacity=0.5,
                      annotation_text=f"Q{p}", annotation_position="top right")

    fig.update_layout(
        title="Flow Duration Curve",
        xaxis_title="Exceedance Probability (%)",
        yaxis_title="Streamflow (mm/day)",
        yaxis_type="log",
        height=400,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def fig_run_comparison(runs_df):
    """Bar chart comparing NSE across all runs with test results."""
    df = runs_df.copy().sort_values("NSE", ascending=True)
    watershed = df["run_name"].str.extract(r"^(lamar|hoh)")[0]
    colors = [WATERSHED_COLORS.get(w, "gray") for w in watershed]

    # Shorten label for display
    short_names = df["run_name"].str.replace(r"_\d{4}_\d{6}$", "", regex=True)
    labels = short_names + "<br>" + df["epoch"]

    fig = go.Figure(go.Bar(
        x=df["NSE"], y=labels,
        orientation="h",
        marker_color=colors,
        text=df["NSE"].round(3),
        textposition="outside",
    ))
    fig.add_vline(x=0.7, line_dash="dash", line_color="green", opacity=0.6,
                  annotation_text="Good", annotation_position="top right")
    fig.update_layout(
        title="NSE Comparison Across All Runs",
        xaxis_title="NSE", height=max(300, 40 * len(df)),
        margin=dict(t=50, l=300, b=40),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("💧 Streamflow Forecast")
st.sidebar.markdown("Transfer Learning Dashboard")

all_runs = discover_runs()

if all_runs.empty:
    st.error("No runs with test results found in `runs/`.")
    st.stop()

# Watershed filter
watersheds = ["All"] + sorted(all_runs["basin"].str.replace("_lumped", "").str.capitalize().unique().tolist())
ws_filter = st.sidebar.selectbox("Watershed", watersheds)

filtered_runs = all_runs.copy()
if ws_filter != "All":
    filtered_runs = filtered_runs[filtered_runs["basin"].str.contains(ws_filter.lower())]

# Run selector
run_options = filtered_runs["run_name"].tolist()

# Default: best NSE run
best_idx = filtered_runs["NSE"].idxmax() if not filtered_runs.empty else 0
default_run = filtered_runs.loc[best_idx, "run_name"] if not filtered_runs.empty else run_options[0]

selected_run = st.sidebar.selectbox(
    "Run",
    run_options,
    index=run_options.index(default_run) if default_run in run_options else 0,
    format_func=lambda x: x,
)

row = filtered_runs[filtered_runs["run_name"] == selected_run].iloc[0]

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Metrics (Test Set)**")
st.sidebar.metric("NSE", f"{row['NSE']:.3f}")
st.sidebar.metric("KGE", f"{row['KGE']:.3f}")
st.sidebar.metric("RMSE", f"{row['RMSE']:.3f} mm/day")
st.sidebar.metric("Pearson-r", f"{row['Pearson-r']:.3f}")


# ── Main area ─────────────────────────────────────────────────────────────────

tab_run, tab_compare, tab_info = st.tabs(["Run Analysis", "Compare Runs", "Model Info"])

# ── Tab 1: Run Analysis ───────────────────────────────────────────────────────
with tab_run:
    df, config, basin = load_run_data(str(row["results_file"]), str(row["config_file"]))
    _, val_df, test_df = split_periods(df, config)

    st.subheader(f"{selected_run}  —  {row['epoch']}")

    # Metric cards
    m = compute_metrics(test_df["obs"].values, test_df["pred"].values)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("NSE", f"{m['NSE']:.3f}")
    c2.metric("KGE (stored)", f"{row['KGE']:.3f}")
    c3.metric("RMSE", f"{m['RMSE']:.3f}")
    c4.metric("MAE", f"{m['MAE']:.3f}")
    c5.metric("Bias", f"{m['Bias']:+.3f}")
    c6.metric("Pearson-r", f"{m['Pearson_r']:.3f}")

    st.markdown("---")

    # Date range slider
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM",
    )
    df_view = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])]

    # Full time series
    st.plotly_chart(
        fig_timeseries(df_view, config, f"Observed vs Predicted — {basin}"),
        use_container_width=True,
    )

    # Test period detail
    val_end = pd.to_datetime(config["validation_end_date"], dayfirst=True)
    test_view = df_view[df_view["date"] > val_end]

    if len(test_view) > 0:
        with st.expander("Test period detail", expanded=True):
            t_obs = test_view["obs"].values
            t_pred = test_view["pred"].values

            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(fig_scatter(t_obs, t_pred, m["R2"]), use_container_width=True)
            with col_r:
                st.plotly_chart(fig_residuals(t_obs, t_pred, m["MAE"]), use_container_width=True)

    # Monthly performance (test period)
    st.markdown("#### Monthly Performance (test period)")
    f_nse, f_rmse = fig_monthly(test_view if len(test_view) > 0 else df_view)
    if f_nse:
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(f_nse, use_container_width=True)
        with col_r:
            st.plotly_chart(f_rmse, use_container_width=True)

    # Flow duration curve
    st.markdown("#### Flow Duration Curve (test period)")
    fdc_data = test_view if len(test_view) > 0 else df_view
    st.plotly_chart(
        fig_flow_duration(fdc_data["obs"].values, fdc_data["pred"].values),
        use_container_width=True,
    )

# ── Tab 2: Compare Runs ───────────────────────────────────────────────────────
with tab_compare:
    st.subheader("All Runs — NSE Comparison")
    st.plotly_chart(fig_run_comparison(all_runs), use_container_width=True)

    st.markdown("---")
    st.subheader("All Test Metrics")
    display_cols = ["run_name", "epoch", "basin", "NSE", "KGE", "RMSE", "Pearson-r"]
    st.dataframe(
        all_runs[display_cols]
        .sort_values("NSE", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            "NSE":      st.column_config.ProgressColumn("NSE",      min_value=0, max_value=1, format="%.3f"),
            "KGE":      st.column_config.ProgressColumn("KGE",      min_value=0, max_value=1, format="%.3f"),
            "Pearson-r":st.column_config.ProgressColumn("Pearson-r",min_value=0, max_value=1, format="%.3f"),
            "RMSE":     st.column_config.NumberColumn("RMSE",       format="%.3f"),
        },
    )

    st.markdown("---")
    st.subheader("Select Runs to Overlay")

    overlay_runs = st.multiselect(
        "Runs to overlay",
        options=all_runs["run_name"].tolist(),
        default=all_runs.sort_values("NSE", ascending=False)["run_name"].head(3).tolist(),
    )

    if overlay_runs:
        fig_overlay = go.Figure()
        first = True
        for rname in overlay_runs:
            rrow = all_runs[all_runs["run_name"] == rname].iloc[0]
            rdf, rcfg, rbasin = load_run_data(str(rrow["results_file"]), str(rrow["config_file"]))
            val_end_r = pd.to_datetime(rcfg["validation_end_date"], dayfirst=True)
            rtest = rdf[rdf["date"] > val_end_r]
            if first:
                fig_overlay.add_trace(go.Scatter(
                    x=rtest["date"], y=rtest["obs"],
                    name="Observed", line=dict(color="black", width=1.5), opacity=0.7,
                ))
                first = False
            fig_overlay.add_trace(go.Scatter(
                x=rtest["date"], y=rtest["pred"],
                name=rname,
                line=dict(width=1.2), opacity=0.85,
            ))

        fig_overlay.update_layout(
            title="Test Period Predictions — Overlay",
            xaxis_title="Date", yaxis_title="Streamflow (mm/day)",
            hovermode="x unified", height=420,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

# ── Tab 3: Model Info ─────────────────────────────────────────────────────────
with tab_info:
    st.subheader(f"Configuration: {selected_run}")
    df_info, cfg, _ = load_run_data(str(row["results_file"]), str(row["config_file"]))

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Architecture**")
        arch_items = {
            "Model": cfg.get("model", "—"),
            "Head": cfg.get("head", "—"),
            "Hidden size": cfg.get("hidden_size", "—"),
            "Output dropout": cfg.get("output_dropout", "—"),
            "Initial forget bias": cfg.get("initial_forget_bias", "—"),
            "Seq length": cfg.get("seq_length", "—"),
            "Predict last n": cfg.get("predict_last_n", "—"),
        }
        st.table(pd.DataFrame(arch_items.items(), columns=["Parameter", "Value"]))

        st.markdown("**Training**")
        train_items = {
            "Optimizer": cfg.get("optimizer", "—"),
            "Loss": cfg.get("loss", "—"),
            "Learning rate": cfg.get("learning_rate", "—"),
            "Batch size": cfg.get("batch_size", "—"),
            "Epochs": cfg.get("epochs", "—"),
        }
        st.table(pd.DataFrame(train_items.items(), columns=["Parameter", "Value"]))

    with col_b:
        st.markdown("**Date Ranges**")
        date_items = {
            "Train start": cfg.get("train_start_date", "—"),
            "Train end": cfg.get("train_end_date", "—"),
            "Val start": cfg.get("validation_start_date", "—"),
            "Val end": cfg.get("validation_end_date", "—"),
            "Test start": cfg.get("test_start_date", "—"),
            "Test end": cfg.get("test_end_date", "—"),
        }
        st.table(pd.DataFrame(date_items.items(), columns=["Period", "Date"]))

        st.markdown("**Variables**")
        st.markdown("*Dynamic inputs:*")
        for v in cfg.get("dynamic_inputs", []):
            st.markdown(f"- `{v}`")
        st.markdown("*Static attributes:*")
        for v in cfg.get("static_attributes", []):
            st.markdown(f"- `{v}`")
        st.markdown("*Target:*")
        for v in cfg.get("target_variables", []):
            st.markdown(f"- `{v}`")

    if cfg.get("finetune_modules"):
        st.info(f"Fine-tuned modules: {', '.join(cfg['finetune_modules'])}")

    if cfg.get("base_run_dir"):
        st.info(f"Base model: `{cfg['base_run_dir']}`")

    st.markdown("---")
    with st.expander("Raw config YAML"):
        st.code(yaml.dump(cfg, default_flow_style=False), language="yaml")
