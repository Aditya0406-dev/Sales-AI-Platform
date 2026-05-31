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
# SYSTEM LOGS AND METRIC STORAGE SETUP
# ==========================================
system_diagnostic_logs = ["🔄 Initializing connection sequence..."]
connection_source = "Local Fail-Safe File Backup"

# ==========================================
# MODULAR FILE INGESTION
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
# BACKEND CORE INTEGRATION (ISOLATED EXECUTION)
# ==========================================
@st.cache_resource
def load_predictive_model():
    global connection_source
    system_diagnostic_logs.append("📦 Attempting to check mlflow framework availability...")
    
    try:
        import mlflow
        import mlflow.xgboost
        system_diagnostic_logs.append("✅ MLflow packages discovered in current runtime.")
        
        experiment = mlflow.get_experiment_by_name("sales_forecasting")
        if experiment is not None:
            system_diagnostic_logs.append(f"📡 Found experiment: '{experiment.name}'")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                latest_run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{latest_run_id}/model"
                system_diagnostic_logs.append(f"📥 Fetching artifact run ID: {latest_run_id}")
                loaded_mlflow_model = mlflow.xgboost.load_model(model_uri)
                return loaded_mlflow_model, f"Live MLflow Server (Run: {latest_run_id[:8]})"
    except Exception as mlflow_error:
        system_diagnostic_logs.append(f"❌ MLflow connection skipped: {str(mlflow_error)}")

    system_diagnostic_logs.append("📂 Reverting back to local backup file validation...")
    if os.path.exists("xgboost_model.json"):
        try:
            local_model = XGBRegressor()
            local_model.load_model("xgboost_model.json")
            system_diagnostic_logs.append("✅ Local backup model parsed successfully.")
            return local_model, "Local Fail-Safe File Backup"
        except Exception as local_err:
            system_diagnostic_logs.append(f"❌ Local structure file corrupted: {str(local_err)}")
            
    return None, None

if not os.path.exists("model_features.json"):
    st.error("Missing configuration file! Run your pipeline script first.")
    st.stop()

with open("model_features.json", "r") as f:
    expected_features = json.load(f)

model, source_info = load_predictive_model()
if source_info is not None:
    connection_source = source_info

if model is None:
    st.error("Critical: Operational model assets missing from server and local storage directories.")
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

        # Generate structural data outputs
        predictions = model.predict(X_inf)
        df_filtered["Predicted_Sales"] = predictions

        total_sales_val = df_filtered["Sales"].sum() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].sum()
        avg_sales_val = df_filtered["Sales"].mean() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].mean()
        highest_sale_val = df_filtered["Sales"].max() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].max()
        lowest_sale_val = df_filtered["Sales"].min() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].min()

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
        # VISIBLE MAIN WORKSPACE WITH DATA SUB-TABS
        # ==========================================
        st.markdown("---")
        st.subheader(f"📊 Workspace Dashboard: {region_filter} ({forecast_type})")
        
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
            
            st.markdown("##### Raw Records Preview Matrix")
            st.dataframe(df_filtered[["Year", "Predicted_Sales"]].head(10), use_container_width=True)
            
        with tab3:
            st.markdown("#### Real-Time Backend Orchestration Environment Audits")
            st.info(f"**Current Connection Resolution Link:** {connection_source}")
            st.markdown("**Server Initialization Output Streams:**")
            for entry_log in system_diagnostic_logs:
                st.code(entry_log)

    else:
        st.warning("⚠️ No dataset records available matching the selected configuration options.")
else:
    st.info("💡 Please upload a clean CSV data spreadsheet file to automatically generate the analytics workspace dashboards.")
