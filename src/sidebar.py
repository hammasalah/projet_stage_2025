import streamlit as st

def render_sidebar(df_data):
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #0059b3, #004080); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("Customer Analysis"):
        st.session_state.page = 'Customer Diagnosis'
        st.rerun()

    if st.sidebar.button("Global Analytics"):
        st.session_state.page = 'Global Analytics'
        st.rerun()

    st.sidebar.markdown("---")

    if st.session_state.page == 'Customer Diagnosis':
        st.sidebar.markdown("## 👤 Customer Selection")
        customer_ids_list = df_data['customerID'].tolist()
        selected_customer_id = st.sidebar.selectbox(
            "🔍 Select Customer ID:",
            customer_ids_list,
            help="Choose a customer ID to analyze their churn probability and risk factors"
        )
        return selected_customer_id
    
    elif st.session_state.page == 'Global Analytics':
        st.sidebar.markdown("## 📊 Global Analytics Filters")
        
        contract_options = ['All'] + df_data['Contract'].unique().tolist()
        selected_contract = st.sidebar.selectbox("Filter by Contract Type", contract_options)

        internet_options = ['All'] + df_data['InternetService'].unique().tolist()
        selected_internet = st.sidebar.selectbox("Filter by Internet Service", internet_options)
        
        payment_options = ['All'] + df_data['PaymentMethod'].unique().tolist()
        selected_payment = st.sidebar.selectbox("Filter by Payment Method", payment_options)
        
        return selected_contract, selected_internet, selected_payment

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="background: #f0f7ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #0059b3;">
        <h4 style="color: #0059b3; margin-top: 0;">ℹ️ About This Dashboard</h4>
        <p style="font-size: 0.9rem; margin-bottom: 0;">
            This AI-powered dashboard provides comprehensive customer churn analysis with explainable predictions to support data-driven retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    return None
