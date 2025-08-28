import streamlit as st
import pandas as pd
import plotly.express as px

def page_global_analytics(df_data):
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
