import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(layout="wide", page_title="AI Sales Forecasting Dashboard")

# ==========================================
# MODULAR FILE INGESTION (HTML/CSS SEPARATION)
# ==========================================
def load_interface_assets():
    if os.path.exists("style.css"):
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            content = f.read()
            clean_content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
            return clean_content.strip()
    return ""

html_template = load_interface_assets()

# ==========================================
# BACKEND CORE INTEGRATION (SAFE MLFLOW CALLS)
# ==========================================
@st.cache_resource
def load_model_safely():
    status_logs = []
    status_logs.append("🔄 Initializing connection sequence...")
    
    # Attempt to safely load MLflow dynamically to bypass top-level import crashes
    try:
        status_logs.append("📦 Attempting to import mlflow framework libraries...")
        import mlflow
        import mlflow.xgboost
        status_logs.append("✅ MLflow packages successfully imported into runtime environment.")
        
        # Look for active training experiment tracking server
        experiment = mlflow.get_experiment_by_name("sales_forecasting")
        if experiment is not None:
            status_logs.append(f"📡 Connected to experiment server: '{experiment.name}' (ID: {experiment.experiment_id})")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                latest_run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{latest_run_id}/model"
                status_logs.append(f"📥 Downloading latest model artifact run: {latest_run_id}")
                loaded_model = mlflow.xgboost.load_model(model_uri)
                status_logs.append("🚀 MLflow Model successfully loaded into active dashboard runtime!")
                return loaded_model, f"Live MLflow Server (Run: {latest_run_id[:8]})", status_logs
            else:
                status_logs.append("⚠️ No active runs found inside the 'sales_forecasting' experiment tracking log.")
        else:
            status_logs.append("ℹ️ Experiment 'sales_forecasting' not registered on active tracking node server.")
    except Exception as mlflow_error:
        status_logs.append(f"❌ MLflow connection handshake bypassed. Context detail: {str(mlflow_error)}")

    # Fallback path if tracking server connection drops out
    status_logs.append("📂 Initializing local storage fail-safe pipeline check...")
    if os.path.exists("xgboost_model.json"):
        try:
            model = XGBRegressor()
            model.load_model("xgboost_model.json")
            status_logs.append("✅ Local file backup model initialized successfully.")
            return model, "Local Fail-Safe File Backup", status_logs
        except Exception as local_error:
            status_logs.append(f"❌ Failed to parse local model structure matrix mapping: {str(local_error)}")
            
    status_logs.append("🚨 Critical Error: No structural predictive asset models discovered anywhere.")
    return None, None, status_logs

if not os.path.exists("model_features.json"):
    st.error("Missing configuration file! Run your pipeline script first.")
    st.stop()

with open("model_features.json", "r") as f:
    expected_features = json.load(f)

# Load tracking framework assets
model, connection_source, system_diagnostic_logs = load_model_safely()
if model is None:
    st.error("Could not load the model artifact asset from server or local path storage.")
    st.stop()

st.sidebar.success(f"Connected Via: {connection_source}")

# ==========================================
# INTERFACE CONTROL ROWS
# ==========================================
uploaded_file = st.file_uploader("Upload CSV File:", type="csv")

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    forecast_type = st.selectbox("Select Forecast Type:", ["Daily Forecast", "Weekly Forecast", "Monthly Forecast"])

with col_ctrl2:
    region_filter = st.selectbox("Select Region:", ["All Regions", "Central", "East", "South", "West"])

with col_ctrl3:
    year_filter = st.selectbox("Select Year:", ["All Years", "2024", "2025", "2026"])

st.caption("Choose forecasting horizon based on business analysis needs.")
generate_btn = st.button("Generate Forecast", type="primary")

# ==========================================
# CORE INFERENCE & RENDERING ENGINE
# ==========================================
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # Safe date compilation layer mapping
    if "Order Date" in df_raw.columns:
        df_raw["Order Date"] = pd.to_datetime(df_raw["Order Date"], errors='coerce')
        df_raw["Year"] = df_raw["Order Date"].dt.year.fillna(2024).astype(int).astype(str)
    elif "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors='coerce')
        df_raw["Year"] = df_raw["Date"].dt.year.fillna(2024).astype(int).astype(str)
    else:
        df_raw["Year"] = "2024"

    df_filtered = df_raw.copy()
    
    if region_filter != "All Regions" and "Region" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Region"] == region_filter]
        
    if year_filter != "All Years" and "Year" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Year"] == year_filter]

    if not df_filtered.empty:
        X_inf = df_filtered.copy()
        if "Sales" in X_inf.columns:
            X_inf = X_inf.drop(columns=["Sales"])
        
        X_inf = X_inf.select_dtypes(include=["int64", "float64", "bool"])
        
        for col in expected_features:
            if col not in X_inf.columns:
                X_inf[col] = 0
        X_inf = X_inf[expected_features]

        try:
            # Run model predictions inference computation
            predictions = model.predict(X_inf)
            df_filtered["Predicted_Sales"] = predictions

            # Calculate aggregated operational business KPIs
            total_sales_val = df_filtered["Sales"].sum() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].sum()
            avg_sales_val = df_filtered["Sales"].mean() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].mean()
            highest_sale_val = df_filtered["Sales"].max() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].max()
            lowest_sale_val = df_filtered["Sales"].min() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].min()

            # Dynamic string formatting inject into the layout HTML template blocks
            if html_template:
                rendered_html = (
                    html_template
                    .replace("{{TOTAL_SALES}}", f"{total_sales_val:,.2f}")
                    .replace("{{AVG_SALES}}", f"{avg_sales_val:,.2f}")
                    .replace("{{HIGHEST_SALE}}", f"{highest_sale_val:,.2f}")
                    .replace("{{LOWEST_SALE}}", f"{lowest_sale_val:,.2f}")
                )
                st.markdown(rendered_html, unsafe_allow_html=True)

            # ==========================================
            # ALWAYS VISIBLE MAIN WORKSPACE WITH DATA SUB-TABS
            # ==========================================
            st.markdown("---")
            st.subheader(f"📊 Workspace Dashboard: {region_filter} ({forecast_type})")
            
            # Sub-tabs layout structure deployment layer
            tab1, tab2, tab3 = st.tabs(["📉 Forecast Trends", "📋 Summary Metrics", "📡 System Logs & MLflow Status"])
            
            with tab1:
                st.markdown("#### Predictive Modeling Timeline View")
                if "Sales" in df_filtered.columns:
                    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
                    ax.set_facecolor('white')

                    split_idx = int(len(df_filtered) * 0.8)
                    hist_slice = df_filtered.iloc[:split_idx]
                    fore_slice = df_filtered.iloc[split_idx:]

                    ax.plot(hist_slice.index, hist_slice["Sales"], label="Historical Sales", color="#0072B2", linewidth=1.5)
                    ax.plot(fore_slice.index, fore_slice["Predicted_Sales"], label="Forecast Predictions", color="#D55E00", linewidth=1.5)

                    ax.set_title(f"{forecast_type} Trend Chart", fontsize=12, fontweight="bold")
                    ax.set_xlabel("Timeline Index")
                    ax.set_ylabel("Sales Volume")
                    ax.grid(True, linestyle='-', color='#CCCCCC', linewidth=0.5)
                    ax.legend(loc="upper right")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close()
                else:
                    st.line_chart(df_filtered["Predicted_Sales"])
            
            with tab2:
                st.markdown("#### Key Filtered Record Highlights")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                col_m1.metric(label="Total Aggregated Sales", value=f"${total_sales_val:,.2f}")
                col_m2.metric(label="Average Order Sales", value=f"${avg_sales_val:,.2f}")
                col_m3.metric(label="Peak Record Transaction", value=f"${highest_sale_val:,.2f}")
                col_m4.metric(label="Minimum Floor Value", value=f"${lowest_sale_val:,.2f}")
