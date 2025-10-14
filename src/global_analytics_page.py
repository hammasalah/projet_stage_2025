import streamlit as st
import pandas as pd
import plotly.express as px

def page_global_analytics(df_data, filters):
    """
    Displays the global analytics page with Power BI-style layout.
    """
    selected_contract, selected_internet, selected_payment = filters
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Global Customer Analytics Dashboard</h1>
        <p>Comprehensive overview of customer base patterns and trends for strategic business insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create a filtered dataframe to be used by all charts
    filtered_df = df_data.copy()
    if selected_contract != 'All':
        filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
    if selected_internet != 'All':
        filtered_df = filtered_df[filtered_df['InternetService'] == selected_internet]
    if selected_payment != 'All':
        filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]

    # Calculate KPIs
    total_customers = filtered_df.shape[0]
    churn_rate = (filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    average_tenure = filtered_df['tenure'].mean()
    avg_monthly_charges = filtered_df['MonthlyCharges'].mean()
    total_revenue = filtered_df['TotalCharges'].sum()

    # --- TOP ROW: KPI Cards (Power BI Style) ---
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #718096; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase;">
                👥 Total Customers
            </h3>
            <div class="metric-value" style="font-size: 2.5rem;">{total_customers:,}</div>
            <p style="color: #48bb78; font-size: 0.875rem; margin-top: 0.5rem;">
                ↑ Active base
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #718096; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase;">
                📊 Churn Rate
            </h3>
            <div class="metric-value" style="font-size: 2.5rem; color: {'#f56565' if churn_rate > 25 else '#48bb78'};">{churn_rate:.1f}%</div>
            <p style="color: {'#f56565' if churn_rate > 25 else '#48bb78'}; font-size: 0.875rem; margin-top: 0.5rem;">
                {'↑ Needs attention' if churn_rate > 25 else '↓ Healthy'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #718096; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase;">
                ⏱️ Avg Tenure
            </h3>
            <div class="metric-value" style="font-size: 2.5rem;">{average_tenure:.1f}</div>
            <p style="color: #4299e1; font-size: 0.875rem; margin-top: 0.5rem;">
                Months
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #718096; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; text-transform: uppercase;">
                💰 Avg Monthly
            </h3>
            <div class="metric-value" style="font-size: 2.5rem;">${avg_monthly_charges:.0f}</div>
            <p style="color: #008080; font-size: 0.875rem; margin-top: 0.5rem;">
                Per customer
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

    # --- SECOND ROW: Main Charts (2 columns) ---
    col1, col2 = st.columns([1.2, 1], gap="medium")

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Churn Rate Trend by Tenure Groups
        filtered_df['TenureGroup'] = pd.cut(filtered_df['tenure'], 
                                            bins=[0, 12, 24, 36, 48, 72], 
                                            labels=['0-12', '12-24', '24-36', '36-48', '48+'])
        churn_by_tenure = filtered_df.groupby('TenureGroup')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        churn_by_tenure.columns = ['Tenure Group', 'Churn Rate']
        
        fig = px.line(
            churn_by_tenure,
            x='Tenure Group',
            y='Churn Rate',
            title="<b>Churn Rate by Tenure Groups</b>",
            markers=True,
            color_discrete_sequence=['#008080']
        )
        fig.update_traces(line=dict(width=4), marker=dict(size=12))
        fig.update_layout(
            font=dict(color='#2d3748', size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            title_font_color='#2d3748',
            height=550,
            xaxis_title="<b>Tenure (Months)</b>",
            yaxis_title="<b>Churn Rate (%)</b>",
            hovermode='x unified',
            xaxis=dict(
                tickfont=dict(size=12, color='#2d3748'),
                title=dict(font=dict(size=14, color='#2d3748'))
            ),
            yaxis=dict(
                tickfont=dict(size=12, color='#2d3748'),
                title=dict(font=dict(size=14, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Contract Distribution
        contract_dist = filtered_df['Contract'].value_counts().reset_index()
        contract_dist.columns = ['Contract', 'Count']
        
        fig = px.pie(
            contract_dist,
            values='Count',
            names='Contract',
            title="<b>Contract Type Distribution</b>",
            color_discrete_sequence=['#008080', '#20b2aa', '#5fc7c7'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textfont_size=18, textfont_color='white')
        fig.update_layout(
            font=dict(color='#2d3748', size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            title_font_color='#2d3748',
            height=550,
            showlegend=True,
            legend=dict(font=dict(size=14))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    # --- THIRD ROW: 3 Medium Charts ---
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Internet Service Distribution
        internet_churn = filtered_df.groupby('InternetService')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        internet_churn.columns = ['Internet Service', 'Churn Rate']
        
        fig = px.bar(
            internet_churn,
            x='Internet Service',
            y='Churn Rate',
            title="<b>Churn by Internet Service</b>",
            color='Churn Rate',
            color_continuous_scale=['#48bb78', '#f56565'],
            text_auto='.1f'
        )
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside', textfont_size=15)
        fig.update_layout(
            font=dict(color='#2d3748', size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_font_color='#2d3748',
            height=500,
            showlegend=False,
            xaxis_title="<b>Service Type</b>",
            yaxis_title="<b>Churn Rate (%)</b>",
            xaxis=dict(
                tickfont=dict(size=13, color='#2d3748'),
                title=dict(font=dict(size=14, color='#2d3748'))
            ),
            yaxis=dict(
                tickfont=dict(size=13, color='#2d3748'),
                title=dict(font=dict(size=14, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Payment Method Churn
        payment_churn = filtered_df.groupby('PaymentMethod')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        payment_churn.columns = ['Payment Method', 'Churn Rate']
        
        fig = px.bar(
            payment_churn,
            x='Payment Method',
            y='Churn Rate',
            title="<b>Churn by Payment Method</b>",
            color='Churn Rate',
            color_continuous_scale=['#48bb78', '#f56565'],
            text_auto='.1f'
        )
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside', textfont_size=15)
        fig.update_layout(
            font=dict(color='#2d3748', size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_font_color='#2d3748',
            height=500,
            showlegend=False,
            xaxis_title="<b>Payment Method</b>",
            yaxis_title="<b>Churn Rate (%)</b>",
            xaxis=dict(
                tickfont=dict(size=12, color='#2d3748'),
                title=dict(font=dict(size=14, color='#2d3748')),
                tickangle=-45
            ),
            yaxis=dict(
                tickfont=dict(size=13, color='#2d3748'),
                title=dict(font=dict(size=14, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Gender Distribution
        gender_dist = filtered_df['gender'].value_counts().reset_index()
        gender_dist.columns = ['Gender', 'Count']
        
        fig = px.pie(
            gender_dist,
            values='Count',
            names='Gender',
            title="<b>Gender Distribution</b>",
            color_discrete_sequence=['#008080', '#20b2aa'],
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16, textfont_color='white')
        fig.update_layout(
            font=dict(color='#2d3748', size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_font_color='#2d3748',
            height=500,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    # --- FOURTH ROW: Large Scatter Plot & Services ---
    col1, col2 = st.columns([1.5, 1], gap="medium")

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Monthly Charges vs Tenure Scatter
        fig = px.scatter(
            filtered_df,
            x='tenure',
            y='MonthlyCharges',
            color='Churn',
            title='<b>Monthly Charges vs. Tenure Analysis</b>',
            labels={'tenure': '<b>Tenure (Months)</b>', 'MonthlyCharges': '<b>Monthly Charges ($)</b>'},
            color_discrete_map={'Yes': '#f56565', 'No': '#48bb78'},
            opacity=0.6,
            size='TotalCharges',
            size_max=15
        )
        fig.update_layout(
            font=dict(color='#2d3748', size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            title_font_color='#2d3748',
            height=550,
            legend=dict(title='<b>Churn Status</b>', font=dict(size=14)),
            xaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(text='<b>Tenure (Months)</b>', font=dict(size=16, color='#2d3748'))
            ),
            yaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(text='<b>Monthly Charges ($)</b>', font=dict(size=16, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Additional Services Adoption
        add_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        service_adoption = []
        for service in add_services:
            adoption_rate = (filtered_df[service] == 'Yes').sum() / len(filtered_df) * 100
            service_adoption.append({'Service': service.replace('Online', ''), 'Adoption': adoption_rate})
        
        service_df = pd.DataFrame(service_adoption)
        fig = px.bar(
            service_df,
            y='Service',
            x='Adoption',
            title="<b>Service Adoption Rates</b>",
            orientation='h',
            color='Adoption',
            color_continuous_scale=['#e0f2f1', '#008080'],
            text_auto='.1f'
        )
        fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside', textfont_size=15)
        fig.update_layout(
            font=dict(color='#2d3748', size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            title_font_color='#2d3748',
            height=550,
            showlegend=False,
            xaxis_title="<b>Adoption Rate (%)</b>",
            yaxis_title="<b>Service Type</b>",
            xaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(font=dict(size=16, color='#2d3748'))
            ),
            yaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(font=dict(size=16, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    # --- FIFTH ROW: Demographics Analysis ---
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Senior Citizen vs Churn
        demographics = []
        demographics.append({
            'Category': 'Senior Citizens',
            'Churn Rate': (filtered_df[filtered_df['SeniorCitizen'] == 1]['Churn'] == 'Yes').sum() / 
                         len(filtered_df[filtered_df['SeniorCitizen'] == 1]) * 100 if len(filtered_df[filtered_df['SeniorCitizen'] == 1]) > 0 else 0
        })
        demographics.append({
            'Category': 'Non-Seniors',
            'Churn Rate': (filtered_df[filtered_df['SeniorCitizen'] == 0]['Churn'] == 'Yes').sum() / 
                         len(filtered_df[filtered_df['SeniorCitizen'] == 0]) * 100 if len(filtered_df[filtered_df['SeniorCitizen'] == 0]) > 0 else 0
        })
        demographics.append({
            'Category': 'With Partner',
            'Churn Rate': (filtered_df[filtered_df['Partner'] == 'Yes']['Churn'] == 'Yes').sum() / 
                         len(filtered_df[filtered_df['Partner'] == 'Yes']) * 100 if len(filtered_df[filtered_df['Partner'] == 'Yes']) > 0 else 0
        })
        demographics.append({
            'Category': 'No Partner',
            'Churn Rate': (filtered_df[filtered_df['Partner'] == 'No']['Churn'] == 'Yes').sum() / 
                         len(filtered_df[filtered_df['Partner'] == 'No']) * 100 if len(filtered_df[filtered_df['Partner'] == 'No']) > 0 else 0
        })
        
        demo_df = pd.DataFrame(demographics)
        fig = px.bar(
            demo_df,
            x='Category',
            y='Churn Rate',
            title="<b>Demographic Churn Analysis</b>",
            color='Category',
            color_discrete_sequence=['#008080', '#20b2aa', '#5fc7c7', '#7dd3d3'],
            text_auto='.1f'
        )
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside', textfont_size=15)
        fig.update_layout(
            font=dict(color='#2d3748', size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            title_font_color='#2d3748',
            height=550,
            showlegend=False,
            xaxis_title="<b>Demographic Category</b>",
            yaxis_title="<b>Churn Rate (%)</b>",
            xaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(font=dict(size=16, color='#2d3748'))
            ),
            yaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(font=dict(size=16, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Tenure Distribution Histogram
        fig = px.histogram(
            filtered_df,
            x='tenure',
            nbins=40,
            title="<b>Customer Tenure Distribution</b>",
            color_discrete_sequence=['#008080'],
            marginal='box'
        )
        fig.update_layout(
            font=dict(color='#2d3748', size=16),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=20,
            title_font_color='#2d3748',
            height=550,
            xaxis_title="<b>Tenure (Months)</b>",
            yaxis_title="<b>Number of Customers</b>",
            showlegend=False,
            xaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(font=dict(size=16, color='#2d3748'))
            ),
            yaxis=dict(
                tickfont=dict(size=14, color='#2d3748'),
                title=dict(font=dict(size=16, color='#2d3748'))
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
