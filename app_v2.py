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
# BACKEND CORE INTEGRATION (FAIL-SAFE LOCAL ONLY)
# ==========================================
@st.cache_resource
def load_model_local():
    if os.path.exists("xgboost_model.json"):
        model = XGBRegressor()
        model.load_model("xgboost_model.json")
        return model, "Local Fail-Safe File Backup"
    return None, None

if not os.path.exists("model_features.json"):
    st.error("Missing configuration file! Run your pipeline script first.")
    st.stop()

with open("model_features.json", "r") as f:
    expected_features = json.load(f)

model, connection_source = load_model_local()
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
generate_btn = st.button("Generate Forecast", type="primary")

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
            # Generate predictions
            predictions = model.predict(X_inf)
            df_filtered["Predicted_Sales"] = predictions

            # Calculate business metrics
            total_sales_val = df_filtered["Sales"].sum() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].sum()
            avg_sales_val = df_filtered["Sales"].mean() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].mean()
            highest_sale_val = df_filtered["Sales"].max() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].max()
            lowest_sale_val = df_filtered["Sales"].min() if "Sales" in df_filtered.columns else df_filtered["Predicted_Sales"].min()

            # Dynamic string mapping insertion straight into the HTML template blocks
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
            # ALWAYS VISIBLE MAIN WORKSPACE WITH SUB-TABS
            # ==========================================
            st.markdown("---")
            st.subheader(f"📊 Workspace Dashboard: {region_filter} ({forecast_type})")
            
            tab1, tab2 = st.tabs(["📉 Forecast Trends", "📋 Summary Metrics"])
            
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
                
                st.markdown("##### Raw Records Preview")
                st.dataframe(df_filtered[["Year", "Predicted_Sales"]].head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error parsing interface metrics: {e}")
    else:
        st.warning("⚠️ No data available matching the selected filter options.")
else:
    st.info("💡 Please upload a CSV file to automatically generate the analytics workspace dashboards.")
