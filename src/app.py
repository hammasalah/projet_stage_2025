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

# --- App setup ---
st.set_page_config(
    page_title="AI Churn Dashboard",
    page_icon="🤖",
    layout="wide"
)

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

# --- Page Functions ---

def page_customer_diagnosis():
    """
    Displays the page for diagnosing a single customer.
    """
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Customer Churn Prediction Dashboard</h1>
        <p>Advanced AI-powered customer analysis to predict churn risk and provide actionable insights for customer retention strategies.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR for user controls ---
    st.sidebar.markdown("---")

    # Creating the dropdown menu to select a customer
    customer_ids_list = df_data['customerID'].tolist()
    selected_customer_id = st.sidebar.selectbox(
        "🔍 Select Customer ID:",
        customer_ids_list,
        help="Choose a customer ID to analyze their churn probability and risk factors"
    )

    st.sidebar.markdown("---")

    # --- MAIN PANEL for displaying results ---
    st.markdown(f"""
    <div class="section-header">
        <h2>📊 Customer Analysis Report</h2>
    </div>
    """, unsafe_allow_html=True)

    # Retrieving the data row for the selected customer
    client_info = df_data[df_data['customerID'] == selected_customer_id]

    # Preprocess the customer data for prediction
    prediction_features = client_info.drop(columns=['customerID', 'Churn'])

    # Make prediction
    churn_probability = model.predict_proba(prediction_features)[0][1]

    # Professional prediction display
    st.markdown(f"""
    <div class="prediction-card">
        <h3 style="color: #0059b3; margin-bottom: 1rem;">Customer ID: {selected_customer_id}</h3>
        <div class="metric-container">
            <div class="metric-value">{churn_probability:.0%}</div>
            <div class="metric-label">Churn Risk Probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    if churn_probability > 0.5:
        st.markdown("""
        <div class="status-high-risk">
            🔴 HIGH RISK: This customer is likely to churn - Immediate intervention recommended
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-loyal">
            🟢 LOW RISK: This customer is likely to remain loyal - Continue current engagement
        </div>
        """, unsafe_allow_html=True)

    # Organizing the display into two columns
    col1, col2 = st.columns([1, 1], gap="large")

    # Left column: Model information
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #0059b3; margin-bottom: 1rem;">🤖 Model Information</h4>
            <p style="color: #2c3e50;"><strong>Algorithm:</strong> CatBoost Gradient Boosting</p>
            <p style="color: #2c3e50;"><strong>Prediction Type:</strong> Binary Classification</p>
            <p style="color: #2c3e50;"><strong>Confidence Level:</strong> High Accuracy Model</p>
            <p style="color: #2c3e50;"><strong>Last Updated:</strong> Current Session</p>
        </div>
        """, unsafe_allow_html=True)

    # Right column: Customer overview
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4 style="color: #0059b3; margin-bottom: 1rem;">👤 Customer Profile</h4>
        """, unsafe_allow_html=True)
        
        # Display key customer metrics in a clean format
        key_metrics = {
            'Gender': client_info['gender'].iloc[0],
            'Contract Type': client_info['Contract'].iloc[0],
            'Monthly Charges': f"${client_info['MonthlyCharges'].iloc[0]:.2f}",
            'Tenure (Months)': client_info['tenure'].iloc[0],
            'Total Charges': f"${client_info['TotalCharges'].iloc[0]:.2f}"
        }
        
        for key, value in key_metrics.items():
            st.markdown(f"<p style='color: #2c3e50;'><strong>{key}:</strong> {value}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Section for XAI explanation
    st.markdown("""
    <div class="section-header">
        <h2>🧠 AI Explainability Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    # Calculate SHAP values for the selected customer
    shap_values = explainer.shap_values(prediction_features)

    # --- Display SHAP Force Plot ---
    st.markdown('<h3 style="color: #0059b3;">📈 Factor Contribution Visualization</h3>', unsafe_allow_html=True)
    
    with st.container():
        shap_plot = shap.force_plot(
            explainer.expected_value,
            shap_values[0, :],
            prediction_features.iloc[0, :],
            matplotlib=True,
            show=False,
            figsize=(20, 6),
            text_rotation=15
        )
        st.pyplot(shap_plot, bbox_inches='tight')
        plt.close()

    # Enhanced explanation section
    st.markdown("""
    <div class="explanation-box">
        <h4 style="color: #0059b3;">📖 How to Interpret This Analysis</h4>
        <p style="color: #2c3e50;">This visualization shows how different customer characteristics influence the churn prediction:</p>
        <ul style="color: #2c3e50;">
            <li><strong>Base Value ({:.2f}):</strong> Average churn probability across all customers</li>
            <li><strong>Red Bars:</strong> Factors that <em>increase</em> churn risk (push right)</li>
            <li><strong>Blue Bars:</strong> Factors that <em>decrease</em> churn risk (push left)</li>
            <li><strong>Final Prediction:</strong> Combined effect of all factors</li>
        </ul>
    </div>
    """.format(explainer.expected_value), unsafe_allow_html=True)

    # --- Generate Enhanced Text-Based Explanation ---
    st.markdown("""
    <div class="section-header">
        <h2>📋 Detailed Risk Factor Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a DataFrame for easier analysis
    feature_names = prediction_features.columns
    feature_values = prediction_features.iloc[0].values
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'value': feature_values,
        'shap_value': shap_values[0]
    })
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values(by='abs_shap', ascending=False)

    # Separate positive and negative contributors
    positive_contributors = shap_df[shap_df['shap_value'] > 0]
    negative_contributors = shap_df[shap_df['shap_value'] < 0]

    base_value = explainer.expected_value
    final_prediction = churn_probability

    col1, col2 = st.columns(2, gap="large")

    with col1:
        if not positive_contributors.empty:
            st.markdown("""
            <div class="info-card" style="border-left-color: #ff4757;">
                <h4 style="color: #ff4757;">⚠️ Risk Increasing Factors</h4>
            """, unsafe_allow_html=True)
            
            for i, (_, row) in enumerate(positive_contributors.head(5).iterrows()):
                impact = "High" if row['abs_shap'] > 0.1 else "Medium" if row['abs_shap'] > 0.05 else "Low"
                st.markdown(f"**{row['feature']}:** {row['value']} *(Impact: {impact})*")
            
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if not negative_contributors.empty:
            st.markdown("""
            <div class="info-card" style="border-left-color: #2ed573;">
                <h4 style="color: #2ed573;">✅ Loyalty Factors</h4>
            """, unsafe_allow_html=True)
            
            for i, (_, row) in enumerate(negative_contributors.head(5).iterrows()):
                impact = "High" if row['abs_shap'] > 0.1 else "Medium" if row['abs_shap'] > 0.05 else "Low"
                st.markdown(f"**{row['feature']}:** {row['value']} *(Impact: {impact})*")
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Summary insights
    st.markdown("""
    <div class="explanation-box">
        <h4 style="color: #0059b3;">🎯 Executive Summary</h4>
        <p style="color: #2c3e50;">Starting from a baseline probability of <strong>{:.1%}</strong>, this customer's specific characteristics 
        result in a final churn prediction of <strong>{:.1%}</strong>.</p>
        <p style="color: #2c3e50;"><strong>Recommendation:</strong> {}</p>
    </div>
    """.format(
        base_value, 
        final_prediction,
        "Focus on retention strategies targeting the key risk factors identified above." if churn_probability > 0.5 
        else "Continue current engagement strategies while monitoring for changes in risk factors."
    ), unsafe_allow_html=True)



def page_global_analytics():
    """
    Displays the global analytics page.
    """
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Global Customer Analytics</h1>
        <p>Comprehensive overview of customer base patterns and trends for strategic business insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Interactive Filters ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Global Analytics Filters")
    
    # Create a filtered dataframe to be used by all charts
    filtered_df = df_data.copy()

    # Filter by Contract Type
    contract_options = ['All'] + df_data['Contract'].unique().tolist()
    selected_contract = st.sidebar.selectbox("Filter by Contract Type", contract_options)
    if selected_contract != 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]

    # Filter by Internet Service
    internet_options = ['All'] + df_data['InternetService'].unique().tolist()
    selected_internet = st.sidebar.selectbox("Filter by Internet Service", internet_options)
    if selected_internet != 'All':
        filtered_df = filtered_df[filtered_df['InternetService'] == selected_internet]
        
    # Filter by Payment Method
    payment_options = ['All'] + df_data['PaymentMethod'].unique().tolist()
    selected_payment = st.sidebar.selectbox("Filter by Payment Method", payment_options)
    if selected_payment != 'All':
        filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]


    # --- Row 1: Key Metrics ---
    st.markdown("""
    <div class="section-header">
        <h2>📊 Key Performance Indicators</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")
    total_customers = filtered_df.shape[0]
    churn_rate = (filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    average_tenure = filtered_df['tenure'].mean()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-container">
                <div class="metric-value">{total_customers:,}</div>
                <div class="metric-label">Total Customers</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-container">
                <div class="metric-value">{churn_rate:.1f}%</div>
                <div class="metric-label">Churn Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-container">
                <div class="metric-value">{average_tenure:.1f}</div>
                <div class="metric-label">Avg. Tenure (Months)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Row 2: Charts ---
    st.markdown("""
    <div class="section-header">
        <h2>📈 Customer Distribution Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")

    with col1:
        fig = px.sunburst(
            filtered_df, 
            path=['Contract', 'Churn'], 
            title="Churn by Contract Type",
            color_discrete_sequence=['#0059b3', '#004080', '#f0f7ff', '#e1ecf4']
        )
        fig.update_layout(
            font=dict(color='#2c3e50', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=16
        )
        fig.update_traces(
            textfont_color='white',
            textfont_size=14,
            insidetextorientation='radial'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_counts = filtered_df['gender'].value_counts().reset_index()
        gender_counts.columns = ['gender', 'count']
        fig = px.pie(
            gender_counts, 
            values='count', 
            names='gender', 
            title="Gender Distribution",
            color_discrete_sequence=['#0059b3', '#004080']
        )
        fig.update_layout(
            font=dict(color='#2c3e50', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=16
        )
        fig.update_traces(
            textinfo='percent+label',
            textposition='inside',
            textfont_size=16,
            textfont_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 3: More Charts ---
    col3, col4 = st.columns(2, gap="large")

    with col3:
        fig = px.histogram(
            filtered_df, 
            x='tenure', 
            nbins=30, 
            title="Customer Tenure Distribution",
            color_discrete_sequence=['#0059b3']
        )
        fig.update_layout(
            font=dict(color='#2c3e50', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=16,
            xaxis_title="Tenure (Months)",
            yaxis_title="Number of Customers",
            xaxis=dict(
                title_font=dict(color='#2c3e50', size=12),
                tickfont=dict(color='#2c3e50', size=11)
            ),
            yaxis=dict(
                title_font=dict(color='#2c3e50', size=12),
                tickfont=dict(color='#2c3e50', size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            filtered_df, 
            x='MonthlyCharges', 
            nbins=30, 
            title="Monthly Charges Distribution",
            color_discrete_sequence=['#0059b3']
        )
        fig.update_layout(
            font=dict(color='#2c3e50', size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=16,
            xaxis_title="Monthly Charges ($)",
            yaxis_title="Number of Customers",
            xaxis=dict(
                title_font=dict(color='#2c3e50', size=12),
                tickfont=dict(color='#2c3e50', size=11)
            ),
            yaxis=dict(
                title_font=dict(color='#2c3e50', size=12),
                tickfont=dict(color='#2c3e50', size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Section 1: Churn Rate by Service Type ---
    st.markdown("""
    <div class="section-header">
        <h2>Churn Rate by Service Type</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Churn by Internet Service
        churn_by_internet = filtered_df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_internet = churn_by_internet.rename(columns={'Yes': 'Churn Rate', 'No': 'Retention Rate'})
        fig = px.bar(
            churn_by_internet, 
            y='Churn Rate',
            title="<b>Churn Rate by Internet Service</b>",
            labels={'Churn Rate': 'Churn Rate (%)', 'InternetService': '<b>Internet Service</b>'},
            text_auto='.0%',
            color_discrete_sequence=['#0059b3']
        )
        fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(
            yaxis_tickformat='.0%',
            font=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            legend_font_size=12
        )
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by Additional Services
        add_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        churn_by_add_service_list = []
        for service in add_services:
            # Consider only customers who have the service
            service_users = filtered_df[filtered_df[service] == 'Yes']
            if not service_users.empty:
                churn_rate = service_users['Churn'].value_counts(normalize=True).get('Yes', 0)
                churn_by_add_service_list.append({'Service': service, 'Churn Rate': churn_rate})
        
        if churn_by_add_service_list:
            churn_by_add_service = pd.DataFrame(churn_by_add_service_list)
            fig = px.bar(
                churn_by_add_service,
                x='Service',
                y='Churn Rate',
                title="<b>Churn Rate for Additional Services</b>",
                labels={'Churn Rate': 'Churn Rate (%)', 'Service': '<b>Service</b>'},
                text_auto='.0%',
                color_discrete_sequence=['#004080']
            )
            fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
            fig.update_layout(
                yaxis_tickformat='.0%',
                font=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#0059b3',
                title_font_size=18,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            fig.update_xaxes(tickfont=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for Additional Services with current filters.")

    # --- Section 2: Financial Analysis ---
    st.markdown("""
    <div class="section-header">
        <h2>Financial Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Monthly Charges vs. Tenure Scatter Plot
        fig = px.scatter(
            filtered_df,
            x='tenure',
            y='MonthlyCharges',
            color='Churn',
            title='<b>Monthly Charges vs. Tenure</b>',
            labels={'tenure': '<b>Tenure (Months)</b>', 'MonthlyCharges': '<b>Monthly Charges ($)</b>'},
            color_discrete_map={'Yes': '#ff4757', 'No': '#2ed573'}
        )
        fig.update_layout(
            font=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            legend_font_size=12
        )
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Churn by Payment Method
        churn_by_payment = filtered_df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_payment = churn_by_payment.rename(columns={'Yes': 'Churn Rate', 'No': 'Retention Rate'})
        fig = px.bar(
            churn_by_payment,
            y='Churn Rate',
            title="<b>Churn Rate by Payment Method</b>",
            labels={'Churn Rate': 'Churn Rate (%)', 'PaymentMethod': '<b>Payment Method</b>'},
            text_auto='.0%',
            color_discrete_sequence=['#0059b3']
        )
        fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(
            yaxis_tickformat='.0%',
            font=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    # --- Section 3: Demographic Insights ---
    st.markdown("""
    <div class="section-header">
        <h2>Demographic Insights</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Churn by Partner and Dependents
        demographics = ['Partner', 'Dependents']
        churn_by_demo_list = []
        for demo in demographics:
            churn_rate_yes = filtered_df[filtered_df[demo] == 'Yes']['Churn'].value_counts(normalize=True).get('Yes', 0)
            churn_by_demo_list.append({'Category': f'Has {demo}', 'Churn Rate': churn_rate_yes})
            churn_rate_no = filtered_df[filtered_df[demo] == 'No']['Churn'].value_counts(normalize=True).get('Yes', 0)
            churn_by_demo_list.append({'Category': f'No {demo}', 'Churn Rate': churn_rate_no})
        
        if churn_by_demo_list:
            churn_by_demo = pd.DataFrame(churn_by_demo_list)
            fig = px.bar(
                churn_by_demo,
                x='Category',
                y='Churn Rate',
                color='Category',
                title="<b>Churn by Demographics</b>",
                labels={'Churn Rate': 'Churn Rate (%)', 'Category': '<b>Demographic Group</b>'},
                text_auto='.0%',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
            fig.update_layout(
                yaxis_tickformat='.0%',
                font=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_color='#0059b3',
                title_font_size=18,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                legend_title_font_size=14,
                legend_font_size=12
            )
            fig.update_xaxes(tickfont=dict(size=14))
            fig.update_yaxes(tickfont=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for Demographics with current filters.")

    with col2:
        # Churn by Senior Citizen
        churn_by_senior = filtered_df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack().fillna(0)
        churn_by_senior = churn_by_senior.rename(columns={'Yes': 'Churn Rate', 'No': 'Retention Rate'})
        churn_by_senior.index = ['Non-Senior', 'Senior']
        fig = px.bar(
            churn_by_senior,
            y='Churn Rate',
            title="<b>Churn Rate by Senior Citizen Status</b>",
            labels={'Churn Rate': 'Churn Rate (%)', 'index': '<b>Senior Citizen Status</b>'},
            text_auto='.0%',
            color_discrete_sequence=['#004080']
        )
        fig.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        fig.update_layout(
            yaxis_tickformat='.0%',
            font=dict(color='#2c3e50', size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='#0059b3',
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        fig.update_xaxes(tickfont=dict(size=14))
        fig.update_yaxes(tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    # --- Row 4: Data Table ---
    st.markdown("""
    <div class="section-header">
        <h2>📋 Sample Customer Data</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p style="color: #2c3e50;">Preview of customer data showing the first 10 records from the dataset based on active filters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        filtered_df.head(10), 
        use_container_width=True,
        height=400
    )

    # Professional call-to-action
    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h4 style="color: #0059b3;">Ready to analyze individual customers?</h4>
            <p>Get detailed predictions and explanations for specific customer churn risk.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Start Customer Analysis", use_container_width=True, type="primary"):
            st.session_state.page = 'Customer Diagnosis'
            st.rerun()

# --- Sidebar Navigation ---
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #0059b3, #004080); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <h2 style="color: white; margin: 0; text-align: center;">🧭 Navigation</h2>
</div>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'Global Analytics'

# Custom navigation buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔍 Customer Analysis", use_container_width=True):
        st.session_state.page = 'Customer Diagnosis'
        st.rerun()

with col2:
    if st.button("📊 Global Analytics", use_container_width=True):
        st.session_state.page = 'Global Analytics'
        st.rerun()

# Add some helpful information in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: #f0f7ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0059b3;">
    <h4 style="color: #0059b3; margin-top: 0;">ℹ️ About This Dashboard</h4>
    <p style="font-size: 0.9rem; margin-bottom: 0;">
        This AI-powered dashboard provides comprehensive customer churn analysis with explainable predictions to support data-driven retention strategies.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Page Routing ---
if st.session_state.page == 'Customer Diagnosis':
    page_customer_diagnosis()
elif st.session_state.page == 'Global Analytics':
    page_global_analytics()
