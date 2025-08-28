import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

def page_customer_diagnosis(df_data, model, explainer, selected_customer_id):
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
