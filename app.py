import streamlit as st
import pandas as pd
import numpy as np
from retirement_simulator import SimulationParams, run_monte_carlo, load_market_data, YearlyRecord, run_sensitivity_analysis, run_retirement_age_comparison
import os
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

st.set_page_config(layout="wide", page_title="Retirement Simulator")

st.title("Retirement Monte Carlo Simulator")
st.write("Plan your financial future with historical market data simulations.")

# Load market data
current_dir = os.path.dirname(__file__)
us_market_data_path = os.path.join(current_dir, 'market_data.csv')
br_market_data_path = os.path.join(current_dir, 'market_data_br.csv')

us_returns = load_market_data(us_market_data_path)
br_returns = load_market_data(br_market_data_path)

# UI for SimulationParams
st.sidebar.header("Simulation Parameters")

with st.sidebar:
    st.subheader("Market Data Source")
    market_source = st.radio("Select Market", ("US (S&P 500)", "Brazil (CDI)"), index=0)

    st.subheader("Personal Details")
    current_age = st.slider("Current Age", 0, 110, 35)
    retirement_age = st.slider("Retirement Age", 55, 85, 65)
    end_age = st.slider("End Age (of simulation)", retirement_age + 1, 110, 95)

    st.subheader("Current Savings (Today's Dollars)")
    pretax_savings = st.number_input("Pre-tax Savings ($)", min_value=0, value=275000, step=10000)
    posttax_savings = st.number_input("Post-tax Savings ($)", min_value=0, value=4800, step=10000)

    st.subheader("Income & Savings")
    salary = st.number_input("Current Annual Salary ($)", min_value=0, value=150000, step=5000)
    salary_growth_rate = st.slider("Real Salary Growth Rate (%)", 0.0, 5.0, 1.0, 0.01) / 100
    pretax_savings_rate = st.slider("Pre-tax Savings Rate (%)", 0.0, 30.0, 14.0, 0.1) / 100
    posttax_savings_rate = st.slider("Post-tax Savings Rate (%)", 0, 30, 5) / 100
    employer_match_rate = st.slider("Employer Match Rate (%)", 0, 100, 100) / 100
    employer_match_cap = st.slider("Employer Match Cap (% of Salary)", 0.0, 10.0, 9.2, 0.1) / 100

    st.subheader("Retirement Spending")
    annual_spending = st.number_input("Desired Annual Spending in Retirement ($)", min_value=0, value=85000, step=1000)

    st.subheader("Simulation Settings")
    num_simulations = st.slider("Number of Monte Carlo Simulations", 100, 5000, 1000, step=100)

# Select the appropriate returns based on user choice
selected_returns = us_returns if market_source == "US (S&P 500)" else br_returns

# Create SimulationParams object
params = SimulationParams(
    current_age=current_age,
    retirement_age=retirement_age,
    end_age=end_age,
    pretax_savings=pretax_savings,
    posttax_savings=posttax_savings,
    salary=salary,
    salary_growth_rate=salary_growth_rate,
    pretax_savings_rate=pretax_savings_rate,
    posttax_savings_rate=posttax_savings_rate,
    employer_match_rate=employer_match_rate,
    employer_match_cap=employer_match_cap,
    annual_spending=annual_spending,
    num_simulations=num_simulations,
    tax_region='BR' if market_source == "Brazil (CDI)" else 'US'
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'figs' not in st.session_state:
    st.session_state['figs'] = None
if 'params' not in st.session_state:
    st.session_state['params'] = None

from pdf_report import generate_pdf_report

# Run Monte Carlo simulation
if st.button("Run Simulation"):
    with st.spinner("Running Monte Carlo simulations..."):
        results = run_monte_carlo(params, selected_returns)
        st.session_state['results'] = results
        st.session_state['params'] = params


if st.session_state['results'] is not None:
    results = st.session_state['results']
    params = st.session_state['params']
    
    st.header("Simulation Summary")

    figs = []

    # --- 1. Main Metrics ---
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results['success_rate'],
            title = {'text': "Success Rate"},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 85], 'color': "gray"}],
                     'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
        gauge_fig.update_layout(height=250)
        st.plotly_chart(gauge_fig, use_container_width=True)
        figs.append(gauge_fig)

    with col2:
        st.metric("5th Percentile", f"${results['percentile_5_final']:,.0f}")
    with col3:
        st.metric("25th Percentile", f"${np.percentile(results['final_balances'], 25):,.0f}")
    with col4:
        st.metric("Median Balance", f"${results['median_final_balance']:,.0f}")
    with col5:
        st.metric("75th Percentile", f"${np.percentile(results['final_balances'], 75):,.0f}")
    
    if results['probability_of_ruin'] > 0:
        st.warning(f"Probability of Ruin: {results['probability_of_ruin']:.1f}%. "
                   f"When ruin occurred, the average shortfall was {results['average_shortfall_years']:.1f} years.")

    st.write("---")
    
    # --- Shared Plotly Config ---
    plotly_config = {
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape', 'resetScale2d']
    }

    # --- 2. New Proactive Visuals ---
    st.header("Analytical Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Final Balances")
        hist_fig = go.Figure(data=[go.Histogram(x=results['final_balances'], nbinsx=50)])
        hist_fig.update_layout(
            title="Histogram of All Possible Final Balances",
            xaxis_title="Final Portfolio Value ($)",
            yaxis_title="Number of Simulations",
        )
        st.plotly_chart(hist_fig, use_container_width=True, config=plotly_config)
        figs.append(hist_fig)

    with col2:
        st.subheader("Portfolio Composition at Retirement")
        comp_labels = ['Initial Savings', 'Total Contributions', 'Market Growth']
        comp_values = [
            params.pretax_savings + params.posttax_savings,
            results['contribution_at_retirement'] - (params.pretax_savings + params.posttax_savings),
            results['growth_at_retirement']
        ]
        donut_fig = go.Figure(data=[go.Pie(labels=comp_labels, values=comp_values, hole=.4)])
        donut_fig.update_layout(title="What Built Your Retirement Nest Egg?")
        st.plotly_chart(donut_fig, use_container_width=True, config=plotly_config)
        figs.append(donut_fig)

    st.write("---")

    # --- 3. Existing Visuals (Improved) ---
    st.header("Portfolio Trajectory & Details")
    
    # Balance Trajectories Chart
    st.subheader("Balance Trajectories (Median and Percentiles)")
    ages = list(results['balance_percentiles'].keys())
    percentiles_df = pd.DataFrame({
        f'{p}th': [results['balance_percentiles'][age][p] for age in ages]
        for p in [5, 25, 50, 75, 95]
    }, index=ages)
    percentiles_df.index.name = 'Age'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentiles_df.index, y=percentiles_df['50th'], mode='lines', name='Median (50th)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=percentiles_df.index, y=percentiles_df['95th'], mode='lines', name='95th', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=percentiles_df.index, y=percentiles_df['75th'], mode='lines', name='75th', line=dict(color='lightgreen', dash='dot')))
    fig.add_trace(go.Scatter(x=percentiles_df.index, y=percentiles_df['25th'], mode='lines', name='25th', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=percentiles_df.index, y=percentiles_df['5th'], mode='lines', name='5th', line=dict(color='red', dash='dot')))
    fig.update_layout(title="Portfolio Balance Over Time (Inflation-Adjusted)", xaxis_title="Age", yaxis_title="Portfolio Balance ($)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)
    figs.append(fig)

    # Survival Probability Chart
    st.subheader("Survival Probability")
    survival_prob_df = pd.DataFrame.from_dict(results['survival_probability'], orient='index', columns=['Survival Probability (%)'])
    survival_prob_df.index.name = 'Age'
    fig_survival = go.Figure(go.Scatter(x=survival_prob_df.index, y=survival_prob_df['Survival Probability (%)'], mode='lines', name='Survival %', fill='tozeroy'))
    fig_survival.update_layout(title="Portfolio Survival Probability Over Time", xaxis_title="Age", yaxis_title="Probability of Not Running Out of Money (%)")
    fig_survival.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_survival, use_container_width=True, config=plotly_config)
    figs.append(fig_survival)

    # --- 4. Cash Flow Table ---
    st.subheader("Median Case: Year-by-Year Cash Flow")
    median_sim_records = results['median_simulation'].yearly_records
    cash_flow_data = []
    for record in median_sim_records:
        cash_flow_data.append({
            'Age': record.age, 'Salary': record.salary, 'Contribution': record.contribution,
            'Employer Match': record.employer_match, 'Withdrawal': record.withdrawal, 'Tax Paid': record.tax_paid,
            'Real Return': record.market_return, 'Pre-tax Balance': record.pretax_balance,
            'Post-tax Balance': record.posttax_balance, 'Total Balance': record.total_balance,
        })
    cash_flow_df = pd.DataFrame(cash_flow_data).set_index('Age')
    cash_flow_df = cash_flow_df.loc[current_age:]
    st.dataframe(cash_flow_df.style.format({
        'Salary': "${:,.0f}", 'Contribution': "${:,.0f}", 'Employer Match': "${:,.0f}",
        'Withdrawal': "${:,.0f}", 'Tax Paid': "${:,.0f}", 'Pre-tax Balance': "${:,.0f}",
        'Post-tax Balance': "${:,.0f}", 'Total Balance': "${:,.0f}", 'Real Return': "{:.2%}"
    }))

    # --- 5. Sensitivity Analysis (Improved) ---
    st.write("---")
    st.header("Sensitivity Analysis")

    with st.spinner("Running sensitivity analyses..."):
        spending_sensitivity = run_sensitivity_analysis(params, selected_returns)
        retirement_age_sensitivity = run_retirement_age_comparison(params, selected_returns)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spending Sensitivity")
        spending_df = pd.DataFrame(spending_sensitivity).T.reset_index()
        spending_df.columns = ['Spending Variation', 'Annual Spending', 'Success Rate', 'Median Final Balance']
        
        fig_spending = go.Figure(go.Bar(
            x=spending_df['Annual Spending'], y=spending_df['Success Rate'],
            text=spending_df['Success Rate'].apply(lambda x: f'{x:.1f}%'), textposition='auto'
        ))
        fig_spending.update_layout(title="Success Rate vs. Annual Spending", xaxis_title="Annual Spending in Retirement ($)", yaxis_title="Success Rate (%)")
        st.plotly_chart(fig_spending, use_container_width=True, config=plotly_config)
        figs.append(fig_spending)

    with col2:
        st.subheader("Retirement Age Sensitivity")
        ret_age_df = pd.DataFrame(retirement_age_sensitivity).T.reset_index()
        ret_age_df.columns = ['Retirement Age', 'Success Rate', 'Median Final Balance', 'p5', 'p95']
        
        # Chart 1: Success Rate vs. Retirement Age
        fig_ret_age_success = go.Figure(go.Bar(
            x=ret_age_df['Retirement Age'], y=ret_age_df['Success Rate'],
            text=ret_age_df['Success Rate'].apply(lambda x: f'{x:.1f}%'), textposition='auto'
        ))
        fig_ret_age_success.update_layout(title="Success Rate vs. Retirement Age", xaxis_title="Retirement Age", yaxis_title="Success Rate (%)")
        st.plotly_chart(fig_ret_age_success, use_container_width=True, config=plotly_config)
        figs.append(fig_ret_age_success)

        # Chart 2: Median Final Balance vs. Retirement Age
        fig_ret_age_balance = go.Figure(go.Bar(
            x=ret_age_df['Retirement Age'], y=ret_age_df['Median Final Balance'],
            text=ret_age_df['Median Final Balance'].apply(lambda x: f'${x:,.0f}'), textposition='auto'
        ))
        fig_ret_age_balance.update_layout(title="Median Final Balance vs. Retirement Age", xaxis_title="Retirement Age", yaxis_title="Median Final Balance ($)")
        st.plotly_chart(fig_ret_age_balance, use_container_width=True, config=plotly_config)
        figs.append(fig_ret_age_balance)

    st.session_state['figs'] = figs

    # --- 6. Simulation Details ---
    st.subheader("Simulation Details")
    st.write(f"Years covered by selected market data: {selected_returns.index.min()} to {selected_returns.index.max()}")
    st.write(f"Mean real return: {selected_returns.mean():.2%}")
    st.write(f"Std dev of real return: {selected_returns.std():.2%}")

    # --- 7. Download PDF ---
    st.write("---")
    st.header("Download Report")

    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF..."):
            pdf_data = generate_pdf_report(st, st.session_state['params'], st.session_state['results'], st.session_state['figs'])
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name="retirement_simulation_report.pdf",
                mime="application/pdf"
            )