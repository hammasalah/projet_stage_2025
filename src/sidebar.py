import streamlit as st

def render_sidebar(df_data):
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 2rem 0;">
        <h2 style="color: #008080; margin: 0; font-weight: 700; font-size: 1.75rem; letter-spacing: -0.02em;">
            📊 Churn Analytics
        </h2>
        <p style="color: #718096; font-size: 0.875rem; margin-top: 0.5rem;">AI-Powered Insights</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <style>
        .nav-button {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            margin-bottom: 8px;
            border-radius: 12px;
            background: transparent;
            color: #718096;
            font-weight: 500;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .nav-button:hover {
            background: linear-gradient(135deg, #008080 0%, #20b2aa 100%);
            color: white;
            transform: translateX(4px);
        }
        .nav-icon {
            font-size: 1.25rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Navigation buttons with icons
    st.sidebar.markdown("### 🧭 Navigation")
    
    if st.sidebar.button("📊  Dashboard Analytics", use_container_width=True):
        st.session_state.page = 'Global Analytics'
        st.rerun()

    if st.sidebar.button("👤  Customer Diagnosis", use_container_width=True):
        st.session_state.page = 'Customer Diagnosis'
        st.rerun()

    st.sidebar.markdown("<div style='margin: 2rem 0; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)

    if st.session_state.page == 'Customer Diagnosis':
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2f1 100%); 
                    padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <h4 style="color: #008080; margin: 0 0 0.5rem 0; font-size: 1rem;">👤 Customer Selection</h4>
        </div>
        """, unsafe_allow_html=True)
        
        customer_ids_list = df_data['customerID'].tolist()
        selected_customer_id = st.sidebar.selectbox(
            "🔍 Select Customer ID:",
            customer_ids_list,
            help="Choose a customer ID to analyze their churn probability and risk factors"
        )
        return selected_customer_id
    
    elif st.session_state.page == 'Global Analytics':
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2f1 100%); 
                    padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <h4 style="color: #008080; margin: 0 0 0.5rem 0; font-size: 1rem;">🎯 Filters</h4>
        </div>
        """, unsafe_allow_html=True)
        
        contract_options = ['All'] + df_data['Contract'].unique().tolist()
        selected_contract = st.sidebar.selectbox("📝 Contract Type", contract_options)

        internet_options = ['All'] + df_data['InternetService'].unique().tolist()
        selected_internet = st.sidebar.selectbox("🌐 Internet Service", internet_options)
        
        payment_options = ['All'] + df_data['PaymentMethod'].unique().tolist()
        selected_payment = st.sidebar.selectbox("💳 Payment Method", payment_options)
        
        return selected_contract, selected_internet, selected_payment

    st.sidebar.markdown("<div style='margin: 2rem 0; border-top: 1px solid #e2e8f0;'></div>", unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2f1 100%); 
                padding: 1.25rem; border-radius: 16px; border-left: 4px solid #008080;">
        <h4 style="color: #008080; margin-top: 0; font-size: 0.95rem; font-weight: 600;">💡 About</h4>
        <p style="font-size: 0.85rem; margin-bottom: 0; color: #2d3748; line-height: 1.6;">
            AI-powered churn analysis with explainable predictions to support data-driven retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return None
