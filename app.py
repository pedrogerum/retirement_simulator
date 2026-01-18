import streamlit as st
import pandas as pd
import numpy as np
from retirement_simulator import SimulationParams, run_monte_carlo, load_market_data, YearlyRecord
import os
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"


def load_historical_events(csv_path: str) -> dict:
    """Load historical events from CSV into dict format."""
    df = pd.read_csv(csv_path)
    events = {}
    for _, row in df.iterrows():
        events[int(row['year'])] = {
            "name": row['name'],
            "color": row['color'],
            "description": row['description']
        }
    return events


def load_mortality_table(csv_path: str) -> dict:
    """Load mortality table from CSV into dict format."""
    df = pd.read_csv(csv_path)
    return {int(row['age']): float(row['qx']) for _, row in df.iterrows()}


# Load data from CSV files
current_dir = os.path.dirname(__file__)
HISTORICAL_EVENTS_US = load_historical_events(os.path.join(current_dir, 'data', 'historical_events_us.csv'))
HISTORICAL_EVENTS_BR = load_historical_events(os.path.join(current_dir, 'data', 'historical_events_br.csv'))
MORTALITY_US = load_mortality_table(os.path.join(current_dir, 'data', 'mortality_us.csv'))
MORTALITY_BR = load_mortality_table(os.path.join(current_dir, 'data', 'mortality_br.csv'))

def calculate_survival_probability(current_age, target_age, mortality_table):
    """Calculate probability of surviving from current_age to target_age."""
    if target_age <= current_age:
        return 1.0

    survival_prob = 1.0
    for age in range(current_age, target_age):
        qx = mortality_table.get(age, 1.0)  # probability of dying at age
        survival_prob *= (1 - qx)  # probability of surviving that year

    return survival_prob

def get_survival_curve(current_age, end_age, mortality_table):
    """Get survival probabilities for each age from current_age to end_age."""
    ages = list(range(current_age, end_age + 1))
    probabilities = [calculate_survival_probability(current_age, age, mortality_table) * 100 for age in ages]
    return ages, probabilities

def calculate_mortality_adjusted_success(mc_results, retirement_age, end_age, mortality_table):
    """Calculate mortality-adjusted success rate from Monte Carlo results."""
    mortality_adjusted = 0.0
    for t in range(retirement_age, end_age):
        survival_to_t = calculate_survival_probability(retirement_age, t, mortality_table)
        q_t = mortality_table.get(t, 1.0)
        prob_death_at_t = survival_to_t * q_t
        portfolio_success_t = mc_results['survival_probability'].get(t, 0) / 100
        mortality_adjusted += prob_death_at_t * portfolio_success_t

    survival_to_end = calculate_survival_probability(retirement_age, end_age, mortality_table)
    portfolio_success_end = mc_results['survival_probability'].get(end_age - 1, 0) / 100
    mortality_adjusted += survival_to_end * portfolio_success_end
    return mortality_adjusted * 100


def run_sensitivity_grid(params, returns, retirement_ages, spending_levels, mortality_table, num_simulations=200):
    """Run Monte Carlo simulations for a grid of retirement ages and spending levels.
    Returns both standard success rate grid and mortality-adjusted success rate grid."""
    from retirement_simulator import SimulationParams, run_monte_carlo

    results_grid = np.zeros((len(spending_levels), len(retirement_ages)))
    mortality_adjusted_grid = np.zeros((len(spending_levels), len(retirement_ages)))

    for i, spending in enumerate(spending_levels):
        for j, ret_age in enumerate(retirement_ages):
            if spending <= 0 or ret_age <= params.current_age or ret_age >= params.end_age:
                results_grid[i, j] = np.nan
                mortality_adjusted_grid[i, j] = np.nan
                continue

            modified_params = SimulationParams(
                current_age=params.current_age,
                retirement_age=ret_age,
                end_age=params.end_age,
                pretax_savings=params.pretax_savings,
                posttax_savings=params.posttax_savings,
                salary=params.salary,
                salary_growth_rate=params.salary_growth_rate,
                pretax_savings_rate=params.pretax_savings_rate,
                posttax_savings_rate=params.posttax_savings_rate,
                employer_match_rate=params.employer_match_rate,
                employer_match_cap=params.employer_match_cap,
                annual_spending=spending,
                num_simulations=num_simulations,
                tax_region=params.tax_region
            )
            mc_results = run_monte_carlo(modified_params, returns)
            results_grid[i, j] = mc_results['success_rate']
            mortality_adjusted_grid[i, j] = calculate_mortality_adjusted_success(
                mc_results, ret_age, params.end_age, mortality_table
            )

    return results_grid, mortality_adjusted_grid

st.set_page_config(layout="wide", page_title="Retirement Simulator")

st.title("Retirement Monte Carlo Simulator")
st.write("Plan your financial future with historical market data simulations.")

# Load market data
us_market_data_path = os.path.join(current_dir, 'data', 'market_data.csv')
br_market_data_path = os.path.join(current_dir, 'data', 'market_data_br.csv')

us_returns = load_market_data(us_market_data_path)
br_returns = load_market_data(br_market_data_path)

# Load full DataFrames for historical events chart
us_market_df = pd.read_csv(us_market_data_path).set_index('Year')
br_market_df = pd.read_csv(br_market_data_path).set_index('Year')

def create_historical_events_chart(market_df, events_dict, market_name):
    """Create an area line chart with historical events annotated."""
    fig = go.Figure()

    # Add area line chart for real returns
    fig.add_trace(go.Scatter(
        x=market_df.index,
        y=market_df['Real_Return'] * 100,
        mode='lines',
        fill='tozeroy',
        name='Real Return',
        line=dict(color='steelblue', width=1),
        fillcolor='rgba(70, 130, 180, 0.3)'
    ))

    # Add vertical dashed lines and markers for events
    for year, event in events_dict.items():
        if year in market_df.index:
            return_value = market_df.loc[year, 'Real_Return'] * 100

            # Vertical dashed line
            fig.add_vline(x=year, line_dash="dash", line_color="gray", opacity=0.5)

            # Scatter point for the event
            fig.add_trace(go.Scatter(
                x=[year],
                y=[return_value],
                mode='markers+text',
                marker=dict(size=12, color=event['color'], symbol='circle'),
                text=[event['name']],
                textposition='top center',
                textfont=dict(size=10),
                name=event['name'],
                showlegend=False
            ))

    fig.update_layout(
        title=f"Historical Real Returns - {market_name}",
        xaxis_title="Year",
        yaxis_title="Real Return (%)",
        hovermode="x unified",
        showlegend=False
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    return fig

def run_stress_test_simulation(starting_balance, annual_spending, years_in_retirement,
                                returns_series, start_year, tax_region='US'):
    """
    Simulate retirement starting from a specific historical year using sequential returns.
    Returns portfolio balance trajectory and ruin info.
    """
    balances = [starting_balance]
    current_balance = starting_balance
    ruin_year = None

    for i in range(years_in_retirement):
        year = start_year + i

        # Get return for this year (wrap around if needed)
        if year in returns_series.index:
            market_return = returns_series[year]
        else:
            # Wrap to beginning of available data
            available_years = returns_series.index.tolist()
            wrapped_idx = i % len(available_years)
            market_return = returns_series.iloc[wrapped_idx]

        # Withdraw spending (simplified - no tax complexity for stress test visualization)
        current_balance -= annual_spending

        # Apply market return
        if current_balance > 0:
            current_balance *= (1 + market_return)

        # Check for ruin
        if current_balance <= 0 and ruin_year is None:
            ruin_year = i + 1  # Years until ruin
            current_balance = 0

        balances.append(max(0, current_balance))

    return {
        'balances': balances,
        'ruin_year': ruin_year,
        'final_balance': balances[-1],
        'survived': ruin_year is None
    }

def create_stress_test_chart(stress_results, events_dict, years_in_retirement):
    """Create a chart showing portfolio trajectories for each historical stress scenario."""
    fig = go.Figure()

    for year, data in stress_results.items():
        event = events_dict[year]
        years = list(range(years_in_retirement + 1))

        # Determine line style based on outcome
        if data['survived']:
            line_style = dict(width=2)
        else:
            line_style = dict(width=2, dash='dot')

        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=years,
            y=data['balances'],
            mode='lines',
            name=f"{year} - {event['name']}",
            line=dict(color=event['color'], **line_style),
            hovertemplate=f"<b>{event['name']} ({year})</b><br>" +
                          "Year %{x}<br>" +
                          "Balance: $%{y:,.0f}<extra></extra>"
        ))

        # Add ruin marker if applicable
        if data['ruin_year'] is not None:
            fig.add_trace(go.Scatter(
                x=[data['ruin_year']],
                y=[0],
                mode='markers',
                marker=dict(size=12, color=event['color'], symbol='x'),
                name=f"Ruin - {event['name']}",
                showlegend=False,
                hovertemplate=f"<b>Portfolio Depleted</b><br>" +
                              f"{event['name']} ({year})<br>" +
                              f"After {data['ruin_year']} years<extra></extra>"
            ))

    fig.update_layout(
        title="Portfolio Stress Test: What If You Retired During These Events?",
        xaxis_title="Years in Retirement",
        yaxis_title="Portfolio Balance ($)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    return fig

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
    # Select mortality table for calculations
    mortality_table = MORTALITY_US if market_source == "US (S&P 500)" else MORTALITY_BR

    # Calculate mortality-adjusted success rate using actuarial approach:
    # Sum over all ages: P(Death at age t | Alive at retirement) × P(Portfolio Success at t)
    # Plus: P(Survive to end_age | Alive at retirement) × P(Portfolio Success at end_age)
    mortality_adjusted_success = 0.0
    for t in range(params.retirement_age, params.end_age):
        # P(Alive at beginning of year t | Alive at retirement)
        # This is the probability of surviving from retirement to age t
        survival_to_t = calculate_survival_probability(params.retirement_age, t, mortality_table)

        # q_t = probability of dying during year t (between age t and t+1)
        q_t = mortality_table.get(t, 1.0)

        # P(Death during year t | Alive at retirement) = P(Survived to t) × q_t
        prob_death_at_t = survival_to_t * q_t

        # P(Portfolio Success at t)
        portfolio_success_t = results['survival_probability'].get(t, 0) / 100

        # Accumulate: weighted portfolio success by probability of dying at that age
        mortality_adjusted_success += prob_death_at_t * portfolio_success_t

    # Add probability of surviving to end_age with successful portfolio
    survival_to_end = calculate_survival_probability(params.retirement_age, params.end_age, mortality_table)
    portfolio_success_end = results['survival_probability'].get(params.end_age - 1, 0) / 100
    mortality_adjusted_success += survival_to_end * portfolio_success_end
    mortality_adjusted_success *= 100  # Convert to percentage

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mortality_adjusted_success,
            title = {'text': "Money Lasts Lifetime"},
            number = {'suffix': '%'},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 50], 'color': "#d73027"},
                         {'range': [50, 85], 'color': "#fee08b"},
                         {'range': [85, 100], 'color': "#1a9850"}],
                     'bar': {'color': "darkblue"}}))
        gauge_fig.update_layout(height=250)
        st.plotly_chart(gauge_fig, use_container_width=True)
        figs.append(gauge_fig)
        st.caption("P(money lasts your lifetime)")

    with col2:
        portfolio_gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = results['success_rate'],
            title = {'text': f"Solvent at {params.end_age}"},
            gauge = {'axis': {'range': [None, 100]},
                     'steps' : [
                         {'range': [0, 50], 'color': "#d73027"},
                         {'range': [50, 85], 'color': "#fee08b"},
                         {'range': [85, 100], 'color': "#1a9850"}],
                     'bar': {'color': "darkblue"}}))
        portfolio_gauge_fig.update_layout(height=250)
        st.plotly_chart(portfolio_gauge_fig, use_container_width=True)
        figs.append(portfolio_gauge_fig)
        st.caption(f"P(portfolio survives to {params.end_age})")

    with col3:
        st.metric("5th Percentile", f"${results['percentile_5_final']:,.0f}")
    with col4:
        st.metric("25th Percentile", f"${np.percentile(results['final_balances'], 25):,.0f}")
    with col5:
        st.metric("Median Balance", f"${results['median_final_balance']:,.0f}")
    with col6:
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

    # --- Mortality-Adjusted Analysis ---
    st.write("---")
    st.subheader("Mortality-Adjusted Analysis")
    st.write("Combines portfolio survival with actuarial mortality data to answer: what's the probability your money lasts as long as you do?")

    # Select mortality table based on market/region (already selected above for metrics)
    mortality_source = "US Social Security Administration 2021" if market_source == "US (S&P 500)" else "Brazil IBGE 2021"

    # Calculate mortality survival curve starting from RETIREMENT AGE (not current age)
    mortality_ages, mortality_probs = get_survival_curve(params.retirement_age, params.end_age, mortality_table)

    # Get portfolio survival probabilities for ages from retirement to end
    portfolio_ages = [age for age in results['survival_probability'].keys() if age >= params.retirement_age]
    portfolio_probs = [results['survival_probability'][age] for age in portfolio_ages]

    # Calculate mortality-adjusted success (portfolio success * probability of being alive)
    # Conditional on being alive at retirement age
    adjusted_success = []
    combined_ages = []
    risk_curve = []  # P(Go broke before death | Alive at age t)
    for age in portfolio_ages:
        if age in range(params.retirement_age, params.end_age + 1):
            # P(Alive at age | Alive at retirement_age)
            mortality_prob = calculate_survival_probability(params.retirement_age, age, mortality_table) * 100
            portfolio_prob = results['survival_probability'].get(age, 100)
            # Mortality-adjusted success: P(portfolio survives AND you're alive | alive at retirement)
            adjusted = (portfolio_prob / 100) * (mortality_prob / 100) * 100
            adjusted_success.append(adjusted)
            combined_ages.append(age)

            # Forward-looking ruin risk: P(Go broke before death | Alive at age t)
            # = Sum over future ages: P(Die at s | Alive at t) × P(Broke by s)
            # + P(Survive to end | Alive at t) × P(Broke by end)
            forward_risk = 0.0

            for future_age in range(age, params.end_age):
                # P(Survive from age to future_age | Alive at age)
                if future_age == age:
                    survival_to_future = 1.0
                else:
                    survival_to_future = calculate_survival_probability(age, future_age, mortality_table)

                # P(Die at future_age | Alive at age) = P(Survive to future_age) × q_future
                q_future = mortality_table.get(future_age, 1.0)
                prob_die_at_future = survival_to_future * q_future

                # P(Broke by future_age) - cumulative from simulations
                portfolio_prob_future = results['survival_probability'].get(future_age, 0)
                prob_broke_by_future = 1 - portfolio_prob_future / 100

                forward_risk += prob_die_at_future * prob_broke_by_future

            # Add survival to end_age term
            survival_to_end = calculate_survival_probability(age, params.end_age, mortality_table)
            prob_broke_by_end = 1 - results['survival_probability'].get(params.end_age - 1, 0) / 100
            forward_risk += survival_to_end * prob_broke_by_end

            risk_curve.append(forward_risk * 100)

    col1, col2 = st.columns(2)

    with col1:
        # Biological survival curve only
        st.markdown("**Mortality Curve**")
        fig_combined = go.Figure()

        # Mortality survival (from retirement age) - green with fill
        fig_combined.add_trace(go.Scatter(
            x=mortality_ages,
            y=mortality_probs,
            mode='lines',
            name='Probability of Being Alive',
            fill='tozeroy',
            line=dict(color='green', width=2),
            fillcolor='rgba(0, 128, 0, 0.2)'
        ))

        # Calculate expected survival age from retirement
        # E[Age] = retirement_age + Σ P(Survive to age t | Alive at retirement)
        expected_age = params.retirement_age
        for t in range(params.retirement_age, params.end_age):
            survival_prob = calculate_survival_probability(params.retirement_age, t + 1, mortality_table)
            expected_age += survival_prob

        # Add annotation for expected survival age
        expected_survival_prob = calculate_survival_probability(params.retirement_age, int(expected_age), mortality_table) * 100
        fig_combined.add_annotation(
            x=expected_age,
            y=expected_survival_prob,
            text=f"Expected: {expected_age:.1f}",
            showarrow=True,
            arrowhead=2
        )

        fig_combined.update_layout(
            title=f"P(Alive at Age t | Alive at {params.retirement_age})",
            xaxis_title="Age",
            yaxis_title="Probability (%)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        fig_combined.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_combined, use_container_width=True, config=plotly_config)
        figs.append(fig_combined)

    with col2:
        # Risk Curve: P(Go Broke Before Death | Alive at t)
        st.markdown("**Lifetime Ruin Risk**")
        fig_risk = go.Figure()

        # Get risk at retirement age for annotation
        retirement_risk = risk_curve[0] if risk_curve else 0

        fig_risk.add_trace(go.Scatter(
            x=combined_ages,
            y=risk_curve,
            mode='lines',
            name='Remaining Ruin Risk',
            fill='tozeroy',
            line=dict(color='crimson', width=2),
            fillcolor='rgba(220, 20, 60, 0.3)',
            hovertemplate="Age %{x}<br>If alive at this age, %{y:.1f}% chance of going broke before death<extra></extra>"
        ))

        # Add annotation for risk at retirement age
        if risk_curve and retirement_risk > 0:
            fig_risk.add_annotation(
                x=params.retirement_age,
                y=retirement_risk,
                text=f"At retirement: {retirement_risk:.1f}%",
                showarrow=True,
                arrowhead=2,
                arrowcolor='crimson',
                font=dict(size=11, color='crimson'),
                bgcolor='white',
                bordercolor='crimson',
                borderwidth=1
            )

        fig_risk.update_layout(
            title="P(Run Out of Money Before Death | Alive at Age t)",
            xaxis_title="Age",
            yaxis_title="Probability (%)",
            hovermode="x unified"
        )
        fig_risk.update_yaxes(range=[0, max(risk_curve) * 1.2 if risk_curve and max(risk_curve) > 0 else 100])
        st.plotly_chart(fig_risk, use_container_width=True, config=plotly_config)
        figs.append(fig_risk)

    # Key insights (conditional on retirement age)
    life_expectancy_prob = calculate_survival_probability(params.retirement_age, params.end_age, mortality_table) * 100
    portfolio_end_prob = results['survival_probability'].get(params.end_age - 1, 0)
    adjusted_end_prob = (portfolio_end_prob / 100) * (life_expectancy_prob / 100) * 100

    # Get risk at retirement for insights
    retirement_risk_info = ""
    if risk_curve and risk_curve[0] > 0:
        retirement_risk_info = f"\n    - **Ruin risk at retirement: {risk_curve[0]:.1f}%** - chance of going broke before death"

    st.info(f"""
    **At Age {params.end_age} (given you reach retirement):**
    - P(portfolio survives): **{portfolio_end_prob:.1f}%**
    - P(you're alive): **{life_expectancy_prob:.1f}%**
    - P(both): **{adjusted_end_prob:.1f}%**{retirement_risk_info}

    **Charts:** The green curve shows P(alive at age t | alive at retirement). The red curve shows P(go broke before death | alive at age t) — this increases over time because surviving longer means you've been consuming savings without dying early.
    """)

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

    # --- 5. Sensitivity Analysis Heatmap ---
    st.write("---")
    st.header("Sensitivity Analysis")
    st.write("How retirement age and spending affect your success probability.")

    # Define grid ranges (7x7 grid for faster computation)
    # Retirement age deltas: -3, -2, -1, 0, +1, +2, +3 years
    age_deltas = [-3, -2, -1, 0, 1, 2, 3]
    retirement_ages = [params.retirement_age + d for d in age_deltas
                       if params.current_age < params.retirement_age + d < params.end_age]
    age_delta_labels = [f"{d:+d}" if d != 0 else "0" for d in age_deltas
                        if params.current_age < params.retirement_age + d < params.end_age]

    # Spending deltas: -30k, -20k, -10k, 0, +10k, +20k, +30k
    spending_deltas = [-30000, -20000, -10000, 0, 10000, 20000, 30000]
    spending_levels = [params.annual_spending + d for d in spending_deltas if params.annual_spending + d > 0]
    spending_delta_labels = [f"{d//1000:+d}k" if d != 0 else "0" for d in spending_deltas if params.annual_spending + d > 0]

    with st.spinner("Running sensitivity grid analysis..."):
        sensitivity_grid, mortality_adjusted_grid = run_sensitivity_grid(
            params, selected_returns, retirement_ages, spending_levels, mortality_table, num_simulations=200
        )

    # Create two columns for side-by-side heatmaps
    heatmap_col1, heatmap_col2 = st.columns(2)

    with heatmap_col1:
        # Create standard success rate heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=sensitivity_grid,
            x=age_delta_labels,
            y=spending_delta_labels,
            colorscale=[
                [0, '#d73027'],      # Red for low success
                [0.5, '#fee08b'],    # Yellow for medium
                [0.85, '#1a9850'],   # Green for high success
                [1, '#006837']       # Dark green for 100%
            ],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in sensitivity_grid],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Retirement Age: %{customdata}<br>Spending: $%{meta:,}<br>Success Rate: %{z:.1f}%<extra></extra>",
            customdata=[[retirement_ages[j] for j in range(len(retirement_ages))] for _ in spending_levels],
            meta=[[spending_levels[i] for _ in retirement_ages] for i in range(len(spending_levels))],
            colorbar=dict(title="Success Rate (%)", ticksuffix="%")
        ))

        # Add marker for current selection (at 0,0)
        if params.retirement_age in retirement_ages and params.annual_spending in spending_levels:
            fig_heatmap.add_trace(go.Scatter(
                x=["0"],
                y=["0"],
                mode='markers',
                marker=dict(size=20, color='white', symbol='circle-open', line=dict(width=3, color='black')),
                name='Current Selection',
                showlegend=True
            ))

        fig_heatmap.update_layout(
            title=f"P(Portfolio Survives to {params.end_age})",
            xaxis_title="Retirement Age (years)",
            yaxis_title="Spending ($ change)",
            height=500
        )

        st.plotly_chart(fig_heatmap, use_container_width=True, config=plotly_config)
        figs.append(fig_heatmap)

    with heatmap_col2:
        # Create mortality-adjusted success rate heatmap
        fig_heatmap_mortality = go.Figure(data=go.Heatmap(
            z=mortality_adjusted_grid,
            x=age_delta_labels,
            y=spending_delta_labels,
            colorscale=[
                [0, '#d73027'],      # Red for low success
                [0.5, '#fee08b'],    # Yellow for medium
                [0.85, '#1a9850'],   # Green for high success
                [1, '#006837']       # Dark green for 100%
            ],
            zmin=0,
            zmax=100,
            text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in mortality_adjusted_grid],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="Retirement Age: %{customdata}<br>Spending: $%{meta:,}<br>Mortality-Adjusted: %{z:.1f}%<extra></extra>",
            customdata=[[retirement_ages[j] for j in range(len(retirement_ages))] for _ in spending_levels],
            meta=[[spending_levels[i] for _ in retirement_ages] for i in range(len(spending_levels))],
            colorbar=dict(title="Success (%)", ticksuffix="%")
        ))

        if params.retirement_age in retirement_ages and params.annual_spending in spending_levels:
            fig_heatmap_mortality.add_trace(go.Scatter(
                x=["0"],
                y=["0"],
                mode='markers',
                marker=dict(size=20, color='white', symbol='circle-open', line=dict(width=3, color='black')),
                name='Current Selection',
                showlegend=True
            ))

        fig_heatmap_mortality.update_layout(
            title="P(Money Lasts Your Lifetime)",
            xaxis_title="Retirement Age (years)",
            yaxis_title="Spending ($ change)",
            height=500
        )

        st.plotly_chart(fig_heatmap_mortality, use_container_width=True, config=plotly_config)
        figs.append(fig_heatmap_mortality)

    # Summary insight
    max_success = np.nanmax(sensitivity_grid)
    max_mortality_adjusted = np.nanmax(mortality_adjusted_grid)
    if max_success < 50:
        st.error("Warning: All scenarios show less than 50% success rate. Consider significantly reducing spending or increasing savings.")
    elif max_success < 80:
        st.warning("Some scenarios may need adjustment. Look for greener areas on the heatmap for better outcomes.")

    st.session_state['figs'] = figs

    # --- 6. Simulation Details ---
    st.subheader("Simulation Details")
    st.write(f"Years covered by selected market data: {selected_returns.index.min()} to {selected_returns.index.max()}")
    st.write(f"Mean real return: {selected_returns.mean():.2%}")
    st.write(f"Std dev of real return: {selected_returns.std():.2%}")

    # --- 7. Historical Market Context ---
    st.write("---")
    with st.expander("Historical Market Context", expanded=False):
        # Select appropriate data and events based on market
        if market_source == "US (S&P 500)":
            market_df = us_market_df
            events_dict = HISTORICAL_EVENTS_US
            market_name = "S&P 500"
        else:
            market_df = br_market_df
            events_dict = HISTORICAL_EVENTS_BR
            market_name = "CDI (Brazil)"

        # Create and display the historical events chart
        fig_historical = create_historical_events_chart(market_df, events_dict, market_name)
        st.plotly_chart(fig_historical, use_container_width=True, config=plotly_config)
        figs.append(fig_historical)

        # Events summary table
        st.subheader("Key Historical Events")
        events_data = []
        for year, event in events_dict.items():
            if year in market_df.index:
                events_data.append({
                    "Year": year,
                    "Event": event['name'],
                    "Real Return": f"{market_df.loc[year, 'Real_Return']:.1%}",
                    "Description": event['description']
                })
        events_table = pd.DataFrame(events_data)
        st.dataframe(events_table, hide_index=True, use_container_width=True)

        # --- Stress Test: Portfolio Performance During Historical Events ---
        st.write("---")
        st.subheader("Stress Test: Your Portfolio During Historical Crises")
        st.write("Simulates your portfolio using actual historical returns from each crisis — what if you retired right before a crash?")

        # Get projected balance at retirement from median simulation
        retirement_idx = params.retirement_age - params.current_age - 1
        if retirement_idx >= 0 and retirement_idx < len(results['median_simulation'].yearly_records):
            projected_retirement_balance = results['median_simulation'].yearly_records[retirement_idx].total_balance
        else:
            projected_retirement_balance = params.pretax_savings + params.posttax_savings

        years_in_retirement = params.end_age - params.retirement_age

        # Run stress tests for each historical event
        stress_results = {}
        for event_year in events_dict.keys():
            if event_year in selected_returns.index:
                stress_results[event_year] = run_stress_test_simulation(
                    starting_balance=projected_retirement_balance,
                    annual_spending=params.annual_spending,
                    years_in_retirement=years_in_retirement,
                    returns_series=selected_returns,
                    start_year=event_year,
                    tax_region=params.tax_region
                )

        # Create and display stress test chart
        if stress_results:
            fig_stress = create_stress_test_chart(stress_results, events_dict, years_in_retirement)
            st.plotly_chart(fig_stress, use_container_width=True, config=plotly_config)
            figs.append(fig_stress)

            # Stress test results table
            st.subheader("Stress Test Results")
            stress_table_data = []
            for year, data in stress_results.items():
                event = events_dict[year]
                if data['survived']:
                    outcome = f"Survived (${data['final_balance']:,.0f} remaining)"
                else:
                    outcome = f"Depleted after {data['ruin_year']} years"

                stress_table_data.append({
                    "Year": year,
                    "Event": event['name'],
                    "Starting Balance": f"${projected_retirement_balance:,.0f}",
                    "Annual Spending": f"${params.annual_spending:,.0f}",
                    "Outcome": outcome
                })

            stress_table = pd.DataFrame(stress_table_data)
            st.dataframe(stress_table, hide_index=True, use_container_width=True)

            # Summary insight
            survivors = sum(1 for d in stress_results.values() if d['survived'])
            total = len(stress_results)
            if survivors == total:
                st.success(f"Your portfolio would have survived all {total} historical stress scenarios.")
            elif survivors == 0:
                st.error(f"Your portfolio would have been depleted in all {total} historical stress scenarios. Consider reducing spending or increasing savings.")
            else:
                st.warning(f"Your portfolio survived {survivors} of {total} historical stress scenarios.")

    # --- 8. Download PDF ---
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