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
from sidebar import render_sidebar

# --- App setup ---
st.set_page_config(
    page_title="AI Churn Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Global Analytics'

# Custom CSS for modern dashboard UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* --- Modern Dashboard Theme --- */
    :root {
        --primary-teal: #008080;
        --teal-light: #20b2aa;
        --teal-dark: #006666;
        --bg-gradient-start: #f0f9ff;
        --bg-gradient-end: #e0f2f1;
        --card-bg: #ffffff;
        --text-primary: #2d3748;
        --text-secondary: #718096;
        --text-muted: #a0aec0;
        --border-subtle: #e2e8f0;
        --shadow-sm: 0 2px 8px rgba(0, 128, 128, 0.08);
        --shadow-md: 0 4px 16px rgba(0, 128, 128, 0.12);
        --shadow-lg: 0 8px 32px rgba(0, 128, 128, 0.16);
        --success-color: #48bb78;
        --danger-color: #f56565;
        --warning-color: #ed8936;
        --info-color: #4299e1;
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* --- Global Styles --- */
    * {
        font-family: var(--font-family);
    }

    body {
        font-family: var(--font-family);
        color: var(--text-primary);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-family);
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* --- Main App Background --- */
    .stApp {
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* --- Modern Sidebar --- */
    [data-testid="stSidebar"] {
        background: var(--card-bg);
        border-right: none;
        box-shadow: 4px 0 20px rgba(0, 128, 128, 0.06);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] h2 {
        color: var(--primary-teal);
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.03em;
    }

    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-weight: 500;
        font-size: 0.9rem;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-primary) !important;
    }

    /* Sidebar navigation buttons */
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 16px;
        padding: 0.75rem 1rem;
        font-weight: 500;
        border: none;
        background: transparent;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        text-align: left;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--teal-light) 100%);
        color: white;
        transform: translateX(4px);
        box-shadow: var(--shadow-md);
    }

    /* --- Top Header Bar --- */
    .dashboard-header {
        background: var(--card-bg);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-sm);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* --- Main Header / Hero Section --- */
    .main-header {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--teal-light) 100%);
        color: white;
        padding: 3rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
    }

    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }

    /* --- Modern Card Design --- */
    .metric-card, .info-card, .prediction-card, .explanation-box {
        background: var(--card-bg);
        border: none;
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover, .info-card:hover, .prediction-card:hover, .explanation-box:hover {
        transform: translateY(-6px);
        box-shadow: var(--shadow-lg);
    }

    .info-card {
        border-left: 4px solid var(--primary-teal);
    }

    .prediction-card {
        text-align: center;
        padding: 2.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    }

    /* --- Summary Cards (Top Row) --- */
    .summary-card {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
    }

    .summary-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
    }

    .summary-card h3 {
        color: var(--text-secondary);
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .summary-card .value {
        color: var(--text-primary);
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .summary-card .trend {
        color: var(--success-color);
        font-size: 0.875rem;
        font-weight: 500;
    }

    .summary-card .trend.down {
        color: var(--danger-color);
    }

    /* --- Section Headers --- */
    .section-header {
        border-bottom: none;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0;
    }

    .section-header h2 {
        color: var(--text-primary) !important;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }

    /* --- Modern Buttons --- */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--teal-light) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        background: linear-gradient(135deg, var(--teal-dark) 0%, var(--primary-teal) 100%);
    }

    /* --- Metric Display --- */
    .metric-container {
        text-align: center;
        padding: 1rem 0;
    }

    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--teal-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.5rem;
    }

    /* --- Status Indicators --- */
    .status-high-risk, .status-loyal {
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-md);
    }

    .status-high-risk {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
    }

    .status-loyal {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    }

    /* --- Chart Containers --- */
    .chart-container {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
    }

    /* --- Text Styling --- */
    .stMarkdown, .stDataFrame {
        color: var(--text-primary) !important;
    }

    .stMarkdown p, .stMarkdown li {
        color: var(--text-primary) !important;
        line-height: 1.7;
    }

    .stMarkdown strong {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    /* --- Streamlit Component Overrides --- */
    .stSelectbox, .stMultiSelect, .stTextInput {
        border-radius: 12px;
    }

    .stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
        border-radius: 12px;
        border-color: var(--border-subtle);
    }

    /* --- Tables --- */
    .dataframe {
        border: none !important;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* --- Progress Indicators --- */
    .stProgress > div > div {
        background: linear-gradient(135deg, var(--primary-teal) 0%, var(--teal-light) 100%);
        border-radius: 10px;
    }

    /* --- Alert Boxes --- */
    .stAlert {
        border-radius: 16px;
        border: none;
        box-shadow: var(--shadow-sm);
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    /* --- Spacing & Layout --- */
    .element-container {
        margin-bottom: 1rem;
    }

    /* --- Smooth Scrollbar --- */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-gradient-start);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-teal);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--teal-dark);
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

# --- Sidebar ---
sidebar_result = render_sidebar(df_data)

# --- Page Routing ---
if st.session_state.page == 'Customer Diagnosis':
    page_customer_diagnosis(df_data, model, explainer, sidebar_result)
elif st.session_state.page == 'Global Analytics':
    page_global_analytics(df_data, sidebar_result)