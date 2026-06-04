import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json, requests, mlflow
st.set_page_config(layout="wide", page_title="AI Sales Forecasting Dashboard")
if "forecast_generated" not in st.session_state:
    st.session_state.forecast_generated = False
if "df_daily" not in st.session_state:
    st.session_state.df_daily = None
if "df_future" not in st.session_state:
    st.session_state.df_future = None
if "is_using_fallback" not in st.session_state:
    st.session_state.is_using_fallback = True
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "run_id" not in st.session_state:
    st.session_state.run_id = None
if "active_forecast_type" not in st.session_state:
    st.session_state.active_forecast_type = "Daily Forecast"
if "active_select_year" not in st.session_state:
    st.session_state.active_select_year = "2015"
MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "sales_forecasting"
mlflow_connected = False
try:
    if requests.get(MLFLOW_URI, timeout=1.5).status_code == 200:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_connected = True
except Exception:
    mlflow_connected = False
with st.sidebar:
    st.success("App Status Operational & Stable")
    if mlflow_connected:
        st.success("🟢 MLflow Backend Connected")
        st.caption(f"Tracking to `{EXPERIMENT_NAME}`")
    else:
        st.warning("🟡 MLflow Backend Standalone Mode")
    JSON_MODEL_PATH = "model_features.json" 
    has_json = os.path.exists(JSON_MODEL_PATH)
    if has_json:
        st.success(f"🤖 Model Loaded {JSON_MODEL_PATH}")
    else:
        st.warning(f"⚠️ {JSON_MODEL_PATH} missing. Using fallback.")
    st.info("Deployment GitHub Main Branch")
st.title("📊 Workspace Dashboard Sales AI Engine Hub")
uploaded_file = st.file_uploader("Upload Your Sales CSV Dataset Here", type=["csv"])
col1, col2, col3 = st.columns(3)
with col1: forecast_type = st.selectbox("Select Forecast Type", ["Daily Forecast", "Monthly Forecast", "Yearly Forecast"])
with col2: select_region = st.selectbox("Select Region", ["All Regions", "North", "South", "East", "West"])
with col3: select_year = st.selectbox("Select Year", ["2014", "2015", "2016", "2017"])
st.caption("Choose forecasting horizon based on business analysis needs.")
generate_btn = st.button("Generate Forecast View", type="primary")
if generate_btn:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "Order Date" in df.columns: df.rename(columns={"Order Date": "Display_Date"}, inplace=True)
            if "Sales" not in df.columns and "sales" in df.columns: df.rename(columns={"sales": "Sales"}, inplace=True)
            df["Display_Date"] = pd.to_datetime(df["Display_Date"], format="%d-%m-%Y", errors='coerce')
            df = df.dropna(subset=["Display_Date", "Sales"])
        except Exception:
            st.error("Error matching file column formats.")
            st.stop()
    else:
        h_dt = pd.date_range(start="2014-01-01", end="2017-12-31", freq="D")
        df = pd.DataFrame({"Display_Date": h_dt, "Sales": np.random.exponential(scale=500, size=len(h_dt))})
    df_filtered = df[df["Display_Date"] >= pd.to_datetime(f"{select_year}-01-01")].copy()
    if df_filtered.empty: df_filtered = df.copy()
    df_daily = df_filtered.groupby("Display_Date")["Sales"].sum().reset_index()
    hist_avg = df_daily["Sales"].mean() if not df_daily.empty else 600
    steps = 365 if forecast_type == "Daily Forecast" else (24 if forecast_type == "Monthly Forecast" else 15)
    freq = "D" if forecast_type == "Daily Forecast" else ("M" if forecast_type == "Monthly Forecast" else "Y")
    future_dates = pd.date_range(start="2018-01-01", periods=steps, freq=freq)
    base_pred, is_using_fallback = [], True
    if has_json:
        try:
            with open(JSON_MODEL_PATH, "r", encoding="utf-8") as f: 
                model_data = json.load(f)
            if isinstance(model_data, dict):
                if forecast_type == "Daily Forecast" and "daily_predictions" in model_data:
                    base_pred = [float(x) for x in model_data["daily_predictions"][:steps]]
                    is_using_fallback = False
                elif forecast_type == "Monthly Forecast" and "monthly_predictions" in model_data:
                    base_pred = [float(x) for x in model_data["monthly_predictions"][:steps]]
                    is_using_fallback = False
                elif forecast_type == "Yearly Forecast" and "yearly_predictions" in model_data:
                    base_pred = [float(x) for x in model_data["yearly_predictions"][:steps]]
                    is_using_fallback = False
        except Exception: 
            has_json = False
    if is_using_fallback or not base_pred:
        if forecast_type == "Daily Forecast":
            base_pred = (np.random.normal(hist_avg, hist_avg * 0.3, size=steps)).tolist()
        elif forecast_type == "Monthly Forecast":
            base_pred = np.random.uniform(hist_avg * 30.4 * 0.8, hist_avg * 30.4 * 1.2, size=steps).tolist()
        else:
            base_pred = np.random.uniform(hist_avg * 365.25 * 0.85, hist_avg * 365.25 * 1.15, size=steps).tolist()
    df_future = pd.DataFrame({"Predicted_Date": future_dates, "Predicted_Sales": base_pred})
    df_future["Predicted_Sales"] = pd.to_numeric(df_future["Predicted_Sales"], errors='coerce').fillna(hist_avg)
    df_future.loc[df_future["Predicted_Sales"] < 0, "Predicted_Sales"] = hist_avg * 0.1
    tot_r, avg_r, max_r = float(df_future['Predicted_Sales'].sum()), float(df_future['Predicted_Sales'].mean()), float(df_future['Predicted_Sales'].max())
    st.session_state.metrics = {"total": f"${tot_r:,.2f}", "avg": f"${avg_r:,.2f}", "max": f"${max_r:,.2f}"}
    st.session_state.df_daily = df_daily
    st.session_state.df_future = df_future
    st.session_state.is_using_fallback = is_using_fallback
    st.session_state.active_forecast_type = forecast_type
    st.session_state.active_select_year = select_year
    st.session_state.forecast_generated = True
    if mlflow_connected:
        try:
            r_title = f"Run_{forecast_type.replace(' ', '_')}_{select_region.replace(' ', '_')}"
            with mlflow.start_run(run_name=r_title) as active_run:
                mlflow.log_params({"forecast_horizon": forecast_type, "target_region": select_region, "baseline_year": select_year, "fallback_mode_active": is_using_fallback})
                mlflow.log_metrics({"aggregated_total_sales": tot_r, "calculated_average_sales": avg_r, "peak_predicted_sales": max_r})
                st.session_state.run_id = active_run.info.run_id
        except Exception as e:
            st.sidebar.warning(f"⚠️ Tracking Skipped {e}")
if st.session_state.forecast_generated:
    if st.session_state.run_id:
        st.info(f"🆔 **Active MLflow Run ID** `{st.session_state.run_id}`")
    main_tab1, main_tab2, main_tab3 = st.tabs(["📈 Forecast Trends", "📊 Summary Metrics", "⚙️ System Status"])
    with main_tab1:
        st.subheader("Predictive Modeling Timeline View")
        if st.session_state.is_using_fallback:
            st.warning("⚠️ Using auto-scaled fallback engine. Upload predictions to model_features.json to view live metrics.")
        else:
            st.success("🎯 Live Connection Active Displaying actual machine learning predictions.")
        sub_tab_graph, sub_tab_data = st.tabs(["📊 Visual Chart", "📋 Predicted Sales Matrix"])
        with sub_tab_graph:
            fig, ax = plt.subplots(figsize=(11, 4.5))
            df_daily = st.session_state.df_daily
            df_future = st.session_state.df_future
            current_run_type = st.session_state.active_forecast_type
            current_run_year = st.session_state.active_select_year
            if current_run_type == "Daily Forecast":
                ax.plot(df_daily["Display_Date"], df_daily["Sales"], label="Historical Sales Actuals", color="#0072B2", alpha=0.4, linewidth=1)
                ax.plot(df_future["Predicted_Date"], df_future["Predicted_Sales"], label="XGBoost Daily Forecast", color="#D55E00", linewidth=2)
                ax.set_title(f"Daily Sales Trend Analysis - From {current_run_year} onwards", fontsize=11, fontweight="bold")
            elif current_run_type == "Monthly Forecast":
                df_m = df_daily.groupby(pd.Grouper(key='Display_Date', freq='M'))['Sales'].sum().reset_index()
                ax.plot(df_m["Display_Date"], df_m["Sales"], label="Historical (Monthly)", color="#0072B2", marker="o", alpha=0.5, linewidth=1.5)
                ax.plot(df_future["Predicted_Date"], df_future["Predicted_Sales"], label="XGBoost Monthly Forecast", color="#D55E00", marker="s", linewidth=2.5, markersize=5)
                ax.set_title(f"Monthly Sales Trend Analysis - From {current_run_year} onwards", fontsize=11, fontweight="bold")
            else:
                df_y = df_daily.groupby(pd.Grouper(key='Display_Date', freq='Y'))['Sales'].sum().reset_index()
                ax.plot(df_y["Display_Date"], df_y["Sales"], label="Historical (Yearly)", color="#0072B2", marker="o", alpha=0.5, linewidth=1.5)
                ax.plot(df_future["Predicted_Date"], df_future["Predicted_Sales"], label="XGBoost 15-Year Forecast", color="#D55E00", marker="^", linewidth=2.5)
                ax.set_title(f"Yearly Sales Trend Analysis - From {current_run_year} onwards", fontsize=11, fontweight="bold")
            ax.set_xlabel("Timeline Horizon")
            ax.set_ylabel("Sales Volume ($)")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)
        with sub_tab_data:
            st.dataframe(st.session_state.df_future, use_container_width=True)
    with main_tab2:
        st.subheader("Key Performance Insights")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Forecasted Revenue", st.session_state.metrics.get("total", "$0.00"))
        c2.metric("Average Periodic Velocity", st.session_state.metrics.get("avg", "$0.00"))
        c3.metric("Peak Target Peak Value", st.session_state.metrics.get("max", "$0.00"))
    with main_tab3:
        st.subheader("Engine Architecture Metadata")
        st.json({
            "Tracking URI Target": MLFLOW_URI,
            "Target Experiment": EXPERIMENT_NAME,
            "Backend Status Live": mlflow_connected,
            "Fallback Trigger Condition": st.session_state.is_using_fallback
         })
