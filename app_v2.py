import sys, subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "--target", "/home/adminuser/venv/lib/python3.10/site-packages", "setuptools"])
import json
import os
import matplotlib.pyplot as plt
import mlflow.xgboost
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
            return f.read()
    return ""

html_template = load_interface_assets()

# ==========================================
# BACKEND CORE INTEGRATION
# ==========================================
@st.cache_resource
def load_model_from_mlflow():
    try:
        experiment = mlflow.get_experiment_by_name("sales_forecasting")
        if experiment is not None:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                latest_run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{latest_run_id}/model"
                return mlflow.xgboost.load_model(model_uri), f"Live MLflow Server (Run: {latest_run_id[:8]})"
    except:
        pass
        
    # FIXED FOR LINUX STABILITY: Native cross-platform loading
    if os.path.exists("xgboost_model.json"):
        try:
            model = XGBRegressor()
            # Explicitly configures the backend booster format to prevent OS path mismatch crashes
            model.load_model("xgboost_model.json")
            return model, "Local Fail-Safe File Backup"
        except:
            pass
    return None, None

if not os.path.exists("model_features.json"):
    st.error("Missing configuration file! Run your pipeline script first.")
    st.stop()

with open("model_features.json", "r") as f:
    expected_features = json.load(f)

model, connection_source = load_model_from_mlflow()
if model is None:
    st.error("Could not load the model artifact.")
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
generate_btn = st.button("Generate Forecast")

# ==========================================
# CORE INFERENCE & RENDERING ENGINE
# ==========================================
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    if "Order Date" in df_raw.columns:
        df_raw["Order Date"] = pd.to_datetime(df_raw["Order Date"])
        df_raw["Year"] = df_raw["Order Date"].dt.year.astype(str)
    elif "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])
        df_raw["Year"] = df_raw["Date"].dt.year.astype(str)

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
            predictions = model.predict(X_inf)
            df_filtered["Predicted_Sales"] = predictions

            total_sales_val = df_filtered["Sales"].sum() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].sum()
            avg_sales_val = df_filtered["Sales"].mean() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].mean()
            highest_sale_val = df_filtered["Sales"].max() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].max()
            lowest_sale_val = df_filtered["Sales"].min() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].min()

            if html_template:
                rendered_html = html_template \
                    .replace("{{TOTAL_SALES}}", f"{total_sales_val:,.2f}") \
                    .replace("{{AVG_SALES}}", f"{avg_sales_val:,.2f}") \
                    .replace("{{HIGHEST_SALE}}", f"{highest_sale_val:,.3f}") \
                    .replace("{{LOWEST_SALE}}", f"{lowest_sale_val:,.2f}")
                
                st.markdown(rendered_html, unsafe_allow_html=True)

            if generate_btn:
                st.markdown("---")
                
                if "Sales" in df_filtered.columns:
                    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
                    ax.set_facecolor('white')

                    split_idx = int(len(df_filtered) * 0.8)
                    hist_slice = df_filtered.iloc[:split_idx]
                    fore_slice = df_filtered.iloc[split_idx:]

                    ax.plot(hist_slice.index, hist_slice["Sales"], label="Historical Sales", color="#0072B2", linewidth=1.5)
                    ax.plot(fore_slice.index, fore_slice["Predicted_Sales"], label="Forecast Sales", color="#D55E00", linewidth=1.5)

                    ax.set_title("Daily Sales Forecast", fontsize=14, fontweight="bold", pad=15)
                    ax.set_xlabel("Date", fontsize=11, labelpad=10)
                    ax.set_ylabel("Sales", fontsize=11, labelpad=10)
                    ax.grid(True, which='both', linestyle='-', color='#CCCCCC', linewidth=0.7)
                    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='#E0E0E0')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    st.pyplot(fig)
                    plt.close()
                else:
                    st.line_chart(df_filtered["Predicted_Sales"])

        except Exception as e:
            st.error(f"Inference processing failed: {e}")
    else:
        st.warning("⚠️ No data available matching the selected filter options.")
