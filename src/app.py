# =============================================================================
# File: src/app.py
# Role: Contains the Streamlit web application for the user interface.
# =============================================================================

# --- 1. Importing necessary libraries ---
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
from customer_diagnosis_page import page_customer_diagnosis
from global_analytics_page import page_global_analytics

# --- App setup ---
st.set_page_config(
    page_title="AI Churn Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Global Analytics'

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #0059b3;
        --secondary-color: #f0f7ff;
        --accent-color: #004080;
        --text-color: #2c3e50;
        --light-bg: #ffffff;
        --border-color: #e1ecf4;
        --page-bg: #fafcff;
    }
    
    /* Main page background */
    .stApp {
        background: linear-gradient(135deg, var(--page-bg) 0%, #f8fbff 100%);
    }
    
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
    }
    
    /* Sidebar background */
    .css-1d391kg, .css-1cypcdb {
        background: linear-gradient(180deg, #f0f7ff 0%, #e8f4ff 100%);
    }
    
    /* Additional background fixes */
    .stApp > header {
        background: transparent;
    }
    
    .stApp [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f7ff 0%, #e8f4ff 100%);
    }
    
    /* Ensure main content has proper background */
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 89, 179, 0.2);
    }
    
    .main-header h1 {
        color: white !important;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Card styling */
    .metric-card {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid var(--border-color);
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 89, 179, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 89, 179, 0.15);
    }
    
    .info-card {
        background: var(--secondary-color);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        color: var(--text-color) !important;
    }
    
    .info-card p, .info-card li, .info-card strong {
        color: var(--text-color) !important;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, var(--light-bg) 0%, var(--secondary-color) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid var(--primary-color);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 89, 179, 0.1);
    }
    
    .section-header {
        background: var(--secondary-color);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 2rem 0 1rem 0;
    }
    
    .section-header h2 {
        color: var(--primary-color) !important;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb {
        background: linear-gradient(180deg, #f0f7ff 0%, #e8f4ff 100%) !important;
    }
    
    .css-1d391kg .css-1v0mbdj {
        background: transparent;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-color);
        border: none;
        color: white;
    }
    
    /* Metric styling */
    .metric-container {
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        font-weight: 500;
    }
    
    /* Status indicators */
    .status-high-risk {
        background: linear-gradient(135deg, #ff4757, #ff3742);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .status-loyal {
        background: linear-gradient(135deg, #2ed573, #1dd1a1);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Table styling */
    .dataframe {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Explanation box */
    .explanation-box {
        background: var(--light-bg);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 89, 179, 0.05);
        color: var(--text-color) !important;
    }
    
    .explanation-box h4 {
        color: var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .explanation-box p, .explanation-box li, .explanation-box ul, .explanation-box strong, .explanation-box em {
        color: var(--text-color) !important;
    }
    
    /* Additional text color fixes */
    div[style*="background: #f0f7ff"] p,
    div[style*="background: #f0f7ff"] h4,
    div[style*="background: #f0f7ff"] strong {
        color: var(--text-color) !important;
    }
    
    /* Ensure all paragraph text is visible */
    .stMarkdown p {
        color: var(--text-color) !important;
    }
    
    /* Fix for dataframe display text */
    .stDataFrame {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Data Loading and Model Loading ---
@st.cache_data
def load_data(path):
    """Loads data from a CSV file."""
    df = pd.read_csv(path)
    # Handle missing TotalCharges for new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    return df

@st.cache_resource
def load_model(path):
    """Loads a pre-trained model."""
    model = joblib.load(path)
    return model

# Load data and model
df_data = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
model = load_model('src/models/catboost_churn_model.joblib')

# --- XAI Setup ---
@st.cache_resource
def get_shap_explainer(_model):
    """Creates a SHAP Tree explainer for the given model."""
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(model)

# --- Page Routing ---
if st.session_state.page == 'Customer Diagnosis':
    page_customer_diagnosis(df_data, model, explainer)
elif st.session_state.page == 'Global Analytics':
    page_global_analytics(df_data)