import streamlit as st
import pandas as pd
import numpy as np
from retirement_simulator import SimulationParams, run_monte_carlo, load_market_data, YearlyRecord, run_sensitivity_grid_fast
from translations import t, h
import os
import plotly.graph_objects as go
import plotly.io as pio
from config import CONFIG

pio.templates.default = "plotly_white"

MAX_AGE = 110

def load_historical_events(region_key: str) -> dict:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, CONFIG[region_key]['historical_events'])
    df = pd.read_csv(csv_path)
    events = {}
    for _, row in df.iterrows():
        events[int(row['year'])] = {
            "name": row['name'],
            "color": row['color'],
            "description": row['description']
        }
    return events

def load_mortality_table(region_key: str) -> dict:
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, CONFIG[region_key]['mortality_table'])
    df = pd.read_csv(csv_path)
    return {int(row['age']): float(row['qx']) for _, row in df.iterrows()}

def load_all_region_data(region_key: str):
    returns_series = load_market_data(region_key)
    current_dir = os.path.dirname(__file__)
    market_data_df_path = os.path.join(current_dir, CONFIG[region_key]['market_data'])
    market_data_df = pd.read_csv(market_data_df_path).set_index('Year')
    historical_events = load_historical_events(region_key)
    mortality_table = load_mortality_table(region_key)
    return {
        'returns_series': returns_series,
        'market_data_df': market_data_df,
        'historical_events': historical_events,
        'mortality_table': mortality_table
    }

def calculate_survival_probability(current_age, target_age, mortality_table):
    if target_age <= current_age: return 1.0
    survival_prob = 1.0
    for age in range(current_age, target_age):
        survival_prob *= (1 - mortality_table.get(age, 1.0))
    return survival_prob

def get_survival_curve(current_age, end_age, mortality_table):
    ages = list(range(current_age, end_age + 1))
    probabilities = [calculate_survival_probability(current_age, age, mortality_table) * 100 for age in ages]
    return ages, probabilities

def calculate_mortality_adjusted_success(mc_results, retirement_age, sim_end_age, mortality_table):
    mortality_adjusted = 0.0
    portfolio_survival_at_end = mc_results['survival_probability'].get(sim_end_age,
        mc_results['survival_probability'].get(sim_end_age - 1, 0)) / 100
    for t in range(retirement_age, MAX_AGE):
        survival_to_t = calculate_survival_probability(retirement_age, t, mortality_table)
        q_t = mortality_table.get(t, 1.0)
        prob_death_at_t = survival_to_t * q_t
        portfolio_success_t = portfolio_survival_at_end if t >= sim_end_age else mc_results['survival_probability'].get(t, 0) / 100
        mortality_adjusted += prob_death_at_t * portfolio_success_t
    survival_to_end = calculate_survival_probability(retirement_age, MAX_AGE, mortality_table)
    mortality_adjusted += survival_to_end * portfolio_survival_at_end
    return mortality_adjusted * 100

def run_sensitivity_grid(params, returns, retirement_ages, spending_levels, mortality_table, num_simulations=200):
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
                current_age=params.current_age, retirement_age=ret_age, end_age=params.end_age,
                pretax_savings=params.pretax_savings, posttax_savings=params.posttax_savings,
                salary=params.salary, salary_growth_rate=params.salary_growth_rate,
                pretax_savings_rate=params.pretax_savings_rate, posttax_savings_rate=params.posttax_savings_rate,
                employer_match_rate=params.employer_match_rate, employer_match_cap=params.employer_match_cap,
                annual_spending=spending, num_simulations=num_simulations, tax_region=params.tax_region
            )
            mc_results = run_monte_carlo(modified_params, returns, mortality_table)
            results_grid[i, j] = mc_results['success_rate']
            mortality_adjusted_grid[i, j] = calculate_mortality_adjusted_success(
                mc_results, ret_age, params.end_age, mortality_table
            )
    return results_grid, mortality_adjusted_grid

def run_stress_test_simulation(starting_balance, annual_spending, years_in_retirement, returns_series, start_year, tax_region='US'):
    balances = [starting_balance]
    current_balance = starting_balance
    ruin_year = None
    for i in range(years_in_retirement):
        year = start_year + i
        if year in returns_series.index:
            market_return = returns_series[year]
        else:
            available_years = returns_series.index.tolist()
            wrapped_idx = i % len(available_years)
            market_return = returns_series.iloc[wrapped_idx]
        current_balance -= annual_spending
        if current_balance > 0:
            current_balance *= (1 + market_return)
        if current_balance <= 0 and ruin_year is None:
            ruin_year = i + 1
            current_balance = 0
        balances.append(max(0, current_balance))
    return {'balances': balances, 'ruin_year': ruin_year, 'final_balance': balances[-1], 'survived': ruin_year is None}

def create_historical_events_chart(market_df, events_dict, market_name, lang="EN"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=market_df.index, y=market_df['Real_Return'] * 100, mode='lines', fill='tozeroy',
        name=t("real_return", lang), line=dict(color='steelblue', width=1), fillcolor='rgba(70, 130, 180, 0.3)'
    ))
    for year, event in events_dict.items():
        if year in market_df.index:
            return_value = market_df.loc[year, 'Real_Return'] * 100
            fig.add_vline(x=year, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_trace(go.Scatter(
                x=[year], y=[return_value], mode='markers+text',
                marker=dict(size=12, color=event['color'], symbol='circle'), text=[event['name']],
                textposition='top center', textfont=dict(size=10), name=event['name'], showlegend=False
            ))
    fig.update_layout(
        title=t("historical_real_returns", lang, market=market_name), xaxis_title=t("year", lang),
        yaxis_title=t("real_return_percent", lang), hovermode="x unified", showlegend=False
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    return fig

def create_stress_test_chart(stress_results, events_dict, years_in_retirement, lang="EN"):
    fig = go.Figure()
    for year, data in stress_results.items():
        event = events_dict[year]
        years = list(range(years_in_retirement + 1))
        line_style = dict(width=2) if data['survived'] else dict(width=2, dash='dot')
        fig.add_trace(go.Scatter(
            x=years, y=data['balances'], mode='lines', name=f"{year} - {event['name']}",
            line=dict(color=event['color'], **line_style),
            hovertemplate=f"<b>{event['name']} ({year})</b><br>" + "Year %{x}<br>" + "Balance: $%{y:,.0f}<extra></extra>"
        ))
        if data['ruin_year'] is not None:
            fig.add_trace(go.Scatter(
                x=[data['ruin_year']], y=[0], mode='markers',
                marker=dict(size=12, color=event['color'], symbol='x'), name=f"Ruin - {event['name']}", showlegend=False,
                hovertemplate=f"<b>Portfolio Depleted</b><br>" + f"{event['name']} ({year})<br>" + f"After {data['ruin_year']} years<extra></extra>"
            ))
    fig.update_layout(
        title=t("stress_test_chart_title", lang), xaxis_title=t("years_in_retirement", lang),
        yaxis_title=t("portfolio_balance", lang), hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    return fig


st.set_page_config(layout="wide", page_title="Retirement Simulator")

def set_language(region_key):
    st.session_state['lang'] = 'PT' if region_key == 'BR' else 'EN'

if 'lang' not in st.session_state:
    st.session_state['lang'] = 'EN'

with st.sidebar:
    st.subheader(t("market_data_source", st.session_state['lang']))
    available_regions = {key: CONFIG[key]['name'] for key in CONFIG.keys()}
    region_display_names = list(available_regions.values())
    default_index = 0
    if 'selected_region_name' in st.session_state:
        try:
            default_index = region_display_names.index(st.session_state['selected_region_name'])
        except ValueError:
            pass
    selected_region_name = st.radio("Market / Mercado", region_display_names, index=default_index, key='market_selector_radio')
    
    st.session_state['selected_region_name'] = selected_region_name
    selected_region_key = next(key for key, name in available_regions.items() if name == selected_region_name)
    set_language(selected_region_key)
    L = st.session_state['lang']
    
    region_data = load_all_region_data(selected_region_key)
    selected_returns = region_data['returns_series']
    selected_market_df = region_data['market_data_df']
    selected_historical_events = region_data['historical_events']
    selected_mortality_table = region_data['mortality_table']

    st.header(t("simulation_parameters", L))
    st.subheader(t("personal_details", L))
    current_age = st.slider(t("current_age", L), 0, MAX_AGE, 35, help=h("current_age", L))
    retirement_age = st.slider(t("retirement_age", L), current_age + 1, MAX_AGE - 1, 65, help=h("retirement_age", L))
    end_age = st.slider(t("end_age", L), retirement_age + 1, MAX_AGE, 95, help=h("end_age", L))
    st.subheader(t("current_savings", L))
    pretax_savings = st.number_input(t("pretax_savings", L), min_value=0, value=275000, step=10000, help=h("pretax_savings", L))
    posttax_savings = st.number_input(t("posttax_savings", L), min_value=0, value=4800, step=10000, help=h("posttax_savings", L))
    st.subheader(t("income_savings", L))
    salary = st.number_input(t("current_salary", L), min_value=0, value=150000, step=5000, help=h("current_salary", L))
    salary_growth_rate = st.slider(t("salary_growth_rate", L), 0.0, 5.0, 1.0, 0.01, help=h("salary_growth_rate", L)) / 100
    pretax_savings_rate = st.slider(t("pretax_savings_rate", L), 0.0, 30.0, 14.0, 0.1, help=h("pretax_savings_rate", L)) / 100
    posttax_savings_rate = st.slider(t("posttax_savings_rate", L), 0, 30, 5, help=h("posttax_savings_rate", L)) / 100
    employer_match_rate = st.slider(t("employer_match_rate", L), 0, 100, 100, help=h("employer_match_rate", L)) / 100
    employer_match_cap = st.slider(t("employer_match_cap", L), 0.0, 10.0, 9.2, 0.1, help=h("employer_match_cap", L)) / 100
    st.subheader(t("retirement_spending", L))
    annual_spending = st.number_input(t("annual_spending", L), min_value=0, value=85000, step=1000, help=h("annual_spending", L))
    st.subheader(t("simulation_settings", L))
    num_simulations = st.slider(t("num_simulations", L), 100, 5000, 1000, step=100, help=h("num_simulations", L))

st.title(t("main_title", L))
st.write(t("main_subtitle", L))

st.info(t("info_simulation_methodology", L, num_simulations=num_simulations))
if selected_region_key == "US":
    st.info(t("info_market_assumptions_us", L))
    st.info(t("info_tax_assumptions_us", L))
    st.info(t("info_mortality_source", L))
else:
    st.info(t("info_market_assumptions_br", L))
    st.info(t("info_tax_assumptions_br", L))
    st.info(t("info_mortality_source_br", L))

params = SimulationParams(
    current_age=current_age, retirement_age=retirement_age, end_age=end_age,
    pretax_savings=pretax_savings, posttax_savings=posttax_savings,
    salary=salary, salary_growth_rate=salary_growth_rate,
    pretax_savings_rate=pretax_savings_rate, posttax_savings_rate=posttax_savings_rate,
    employer_match_rate=employer_match_rate, employer_match_cap=employer_match_cap,
    annual_spending=annual_spending, num_simulations=num_simulations, tax_region=selected_region_key
)

from pdf_report import generate_pdf_report

if st.button(t("run_simulation", L)):
    with st.spinner(t("running_simulations", L)):
        st.session_state['results'] = run_monte_carlo(params, selected_returns, selected_mortality_table)
        st.session_state['params'] = params

if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']
    params = st.session_state['params']

    st.header(t("simulation_summary", L))

    figs = []

    mortality_adjusted_success = calculate_mortality_adjusted_success(results, params.retirement_age, params.end_age, selected_mortality_table)

    col1, col2 = st.columns(2)
    with col1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=mortality_adjusted_success,
            title={'text': t("money_lasts_lifetime", L)}, number={'suffix': '%'},
            gauge={'axis': {'range': [None, 100]}, 'steps': [{'range': [0, 50], 'color': "#d73027"}, {'range': [50, 85], 'color': "#fee08b"}, {'range': [85, 100], 'color': "#1a9850"}], 'bar': {'color': "darkblue"}}
        )).update_layout(height=250, margin=dict(t=40, b=40))
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.caption(h("money_lasts_lifetime", L))
        figs.append(gauge_fig)
        
        st.subheader(t("money_left_at_death_title", L))
        percentiles_to_show = [5, 25, 50, 75, 95]
        percentile_values_death = [np.percentile(results['balances_at_death'], p) for p in percentiles_to_show]
        percentile_labels = [t(f"percentile_{p}th_label", L) for p in percentiles_to_show]
        bar_fig_death = go.Figure(go.Bar(
            x=percentile_labels, y=percentile_values_death, text=[f"${v:,.0f}" for v in percentile_values_death],
            textposition='auto', marker_color=['#d73027', '#fee08b', '#1a9850', '#1a9850', '#1a9850']
        )).update_layout(title_text=t("range_of_outcomes_title_death", L), xaxis_title=t("scenario_outcome", L), yaxis_title=t("final_portfolio_value", L))
        st.plotly_chart(bar_fig_death, use_container_width=True)
        st.info(t("info_bar_chart_death", L))
        figs.append(bar_fig_death)

    with col2:
        portfolio_gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number", value=results['success_rate'],
            title={'text': t("money_left_at_age_gauge", L, age=params.end_age)}, number={'suffix': '%'},
            gauge={'axis': {'range': [None, 100]}, 'steps': [{'range': [0, 50], 'color': "#d73027"}, {'range': [50, 85], 'color': "#fee08b"}, {'range': [85, 100], 'color': "#1a9850"}], 'bar': {'color': "darkblue"}}
        )).update_layout(height=250, margin=dict(t=40, b=40))
        st.plotly_chart(portfolio_gauge_fig, use_container_width=True)
        st.caption(h("solvent_at_age", L, age=params.end_age, num_simulations=params.num_simulations))
        figs.append(portfolio_gauge_fig)
        
        st.subheader(t("money_left_at_age", L, age=params.end_age))
        percentile_values_end = [np.percentile(results['final_balances'], p) for p in percentiles_to_show]
        bar_fig_end = go.Figure(go.Bar(
            x=percentile_labels, y=percentile_values_end, text=[f"${v:,.0f}" for v in percentile_values_end],
            textposition='auto', marker_color=['#d73027', '#fee08b', '#1a9850', '#1a9850', '#1a9850']
        )).update_layout(title_text=t("range_of_outcomes_title", L, end_age=params.end_age), xaxis_title=t("scenario_outcome", L), yaxis_title=t("final_portfolio_value", L))
        st.plotly_chart(bar_fig_end, use_container_width=True)
        st.info(t("info_bar_chart_end_age", L, end_age=params.end_age))
        figs.append(bar_fig_end)

    if results['probability_of_ruin'] > 0:
        lifetime_ruin_prob = 100 - mortality_adjusted_success
        st.warning(t("ruin_warning", L,
                     lifetime_prob=lifetime_ruin_prob,
                     age_prob=results['probability_of_ruin'],
                     end_age=params.end_age))
    
    st.write("---")
    st.header(t("portfolio_trajectory", L))

    st.subheader(t("balance_trajectories", L), help=h("balance_trajectories", L, num_simulations=params.num_simulations))
    ages = list(results['balance_percentiles'].keys())
    percentiles_df = pd.DataFrame({
        f'{p}th': [results['balance_percentiles'][age][p] for age in ages]
        for p in [5, 25, 50, 75, 95]
    }, index=ages)
    percentiles_df.index.name = 'Age'
    fig_trajectories = go.Figure()
    final_age_traj = ages[-1]
    final_values_traj = {p: percentiles_df[f'{p}th'].iloc[-1] for p in [5, 25, 50, 75, 95]}
    for p_val, color, dash in [(50, 'blue', None), (95, 'green', 'dot'), (75, 'lightgreen', 'dot'), (25, 'orange', 'dot'), (5, 'red', 'dot')]:
        fig_trajectories.add_trace(go.Scatter(x=percentiles_df.index, y=percentiles_df[f'{p_val}th'], mode='lines', name=t(f"percentile_{p_val}th_legend", L), line=dict(color=color, width=4 if p_val==50 else 2, dash=dash)))
    # Add retirement age marker
    if params.retirement_age in percentiles_df.index:
        fig_trajectories.add_vline(x=params.retirement_age, line_dash="dash", line_color="gray", opacity=0.7)
        fig_trajectories.add_annotation(x=params.retirement_age, y=1, yref="paper", text="Retirement", showarrow=False, yanchor="bottom")
    fig_trajectories.update_layout(title=t("portfolio_balance_title", L), xaxis_title=t("age", L), yaxis_title=t("portfolio_balance", L), hovermode="x unified", margin=dict(r=100))
    st.plotly_chart(fig_trajectories, use_container_width=True)
    figs.append(fig_trajectories)

    st.write("---")
    
    col1_retirement, col2_donut = st.columns(2)
    with col1_retirement:
        st.subheader(t("money_at_retirement_title", L, retirement_age=params.retirement_age))
        if params.retirement_age in results['balance_percentiles']:
            percentile_values_retirement = [results['balance_percentiles'][params.retirement_age][p] for p in percentiles_to_show]
            bar_fig_retirement = go.Figure(go.Bar(
                x=percentile_labels, y=percentile_values_retirement,
                text=[f"${v:,.0f}" for v in percentile_values_retirement],
                textposition='auto', marker_color=['#d73027', '#fee08b', '#1a9850', '#1a9850', '#1a9850']
            ))
            bar_fig_retirement.update_layout(
                title_text=t("range_of_outcomes_retirement", L, retirement_age=params.retirement_age),
                xaxis_title=t("scenario_outcome", L), yaxis_title=t("portfolio_balance", L)
            )
            st.plotly_chart(bar_fig_retirement, use_container_width=True)
            st.info(t("info_bar_chart_retirement", L, retirement_age=params.retirement_age))
            figs.append(bar_fig_retirement)
    
    with col2_donut:
        st.subheader(t("portfolio_composition", L), help=h("portfolio_composition", L))
        comp_labels = [t("initial_savings", L), t("total_contributions", L), t("market_growth", L)]
        comp_values = [
            params.pretax_savings + params.posttax_savings,
            results['contribution_at_retirement'] - (params.pretax_savings + params.posttax_savings),
            results['growth_at_retirement']
        ]
        donut_fig = go.Figure(data=[go.Pie(labels=comp_labels, values=comp_values, hole=.4)])
        donut_fig.update_layout(title=t("what_built_nest_egg", L))
        st.plotly_chart(donut_fig, use_container_width=True)
        figs.append(donut_fig)

    st.write("---")
    st.subheader(t("mortality_adjusted_analysis", L))
    st.write(t("mortality_description", L))

    mortality_ages, mortality_probs = get_survival_curve(params.retirement_age, MAX_AGE, selected_mortality_table)
    portfolio_survival_at_end = results['survival_probability'].get(params.end_age - 1, 0)
    extended_portfolio_probs = {}
    for age in range(params.retirement_age, MAX_AGE + 1):
        extended_portfolio_probs[age] = results['survival_probability'].get(age, portfolio_survival_at_end)

    risk_curve = []
    combined_ages = list(range(params.retirement_age, MAX_AGE + 1))
    for age in combined_ages:
        forward_risk = 0.0
        for future_age in range(age, MAX_AGE):
            survival_to_future = calculate_survival_probability(age, future_age, selected_mortality_table)
            q_future = selected_mortality_table.get(future_age, 1.0)
            prob_die_at_future = survival_to_future * q_future
            prob_broke_by_future = 1 - (extended_portfolio_probs.get(future_age, 0) / 100)
            forward_risk += prob_die_at_future * prob_broke_by_future
        if age < MAX_AGE:
            survival_to_max_age = calculate_survival_probability(age, MAX_AGE, selected_mortality_table)
            prob_broke_by_max_age = 1 - (extended_portfolio_probs.get(MAX_AGE, 0) / 100)
            forward_risk += survival_to_max_age * prob_broke_by_max_age
        risk_curve.append(forward_risk * 100)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t('mortality_curve', L)}**", help=h("mortality_curve", L))
        fig_combined = go.Figure(go.Scatter(
            x=mortality_ages, y=mortality_probs, mode='lines', name=t("prob_being_alive", L),
            fill='tozeroy', line=dict(color='green', width=2), fillcolor='rgba(0, 128, 0, 0.2)'
        ))
        # Calculate life expectancy (expected age) - the average age at death for someone currently this age
        life_expectancy = params.retirement_age + sum(calculate_survival_probability(params.retirement_age, age + 1, selected_mortality_table) for age in range(params.retirement_age, MAX_AGE))
        fig_combined.add_annotation(x=life_expectancy, y=calculate_survival_probability(params.retirement_age, int(life_expectancy), selected_mortality_table) * 100, text=f"Life Expectancy: {life_expectancy:.1f}", showarrow=True, arrowhead=2)
        fig_combined.update_layout(title=t("alive_at_age_title", L, age=params.retirement_age), xaxis_title=t("age", L), yaxis_title=t("probability_percent", L), hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        fig_combined.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_combined, use_container_width=True)
        st.info(t("info_mortality_curve", L, life_expectancy=life_expectancy))

    with col2:
        st.markdown(f"**{t('lifetime_ruin_risk', L)}**", help=h("lifetime_ruin_risk", L))
        fig_risk = go.Figure(go.Scatter(
            x=combined_ages, y=risk_curve, mode='lines', name=t("remaining_ruin_risk", L),
            fill='tozeroy', line=dict(color='crimson', width=2), fillcolor='rgba(220, 20, 60, 0.3)',
            hovertemplate="Age %{x}<br>%{y:.1f}%<extra></extra>"
        ))
        retirement_risk = risk_curve[0] if risk_curve else 0
        if risk_curve and retirement_risk > 0:
            fig_risk.add_annotation(x=params.retirement_age, y=retirement_risk, text=t("at_retirement", L, value=retirement_risk), showarrow=True, arrowhead=2, arrowcolor='crimson', font=dict(size=11, color='crimson'), bgcolor='white', bordercolor='crimson', borderwidth=1)
        fig_risk.update_layout(title=t("ruin_risk_title", L), xaxis_title=t("age", L), yaxis_title=t("probability_percent", L), hovermode="x unified")
        fig_risk.update_yaxes(range=[0, max(risk_curve) * 1.2 if risk_curve and max(risk_curve) > 0 else 100])
        st.plotly_chart(fig_risk, use_container_width=True)
        figs.append(fig_risk)
        st.info(t("info_ruin_risk_curve", L, retirement_risk=retirement_risk))
    
    st.subheader(t("cash_flow_table", L))
    median_sim_records = results['median_simulation'].yearly_records
    cash_flow_data = []
    for record in median_sim_records:
        cash_flow_data.append({
            t("age", L): record.age, t("salary", L): record.salary, t("contribution", L): record.contribution,
            t("employer_match", L): record.employer_match, t("withdrawal", L): record.withdrawal, t("tax_paid", L): record.tax_paid,
            t("real_return", L): record.market_return, t("pretax_balance", L): record.pretax_balance,
            t("posttax_balance", L): record.posttax_balance, t("total_balance", L): record.total_balance,
        })
    cash_flow_df = pd.DataFrame(cash_flow_data).set_index(t("age", L))
    cash_flow_df = cash_flow_df.loc[current_age:]
    st.dataframe(cash_flow_df.style.format({
        t("salary", L): "${:,.0f}", t("contribution", L): "${:,.0f}", t("employer_match", L): "${:,.0f}",
        t("withdrawal", L): "${:,.0f}", t("tax_paid", L): "${:,.0f}", t("pretax_balance", L): "${:,.0f}",
        t("posttax_balance", L): "${:,.0f}", t("total_balance", L): "${:,.0f}", t("real_return", L): "{:.2%}"
    }))

    st.write("---")
    st.header(t("sensitivity_analysis", L), help=h("sensitivity_analysis", L))
    st.write(t("sensitivity_description", L))

    age_deltas = [-3, -2, -1, 0, 1, 2, 3]
    retirement_ages_sens = [params.retirement_age + d for d in age_deltas if params.current_age < params.retirement_age + d < params.end_age]
    age_delta_labels = [f"{d:+d}" if d != 0 else "0" for d in age_deltas if params.current_age < params.retirement_age + d < params.end_age]
    spending_deltas = [-30000, -20000, -10000, 0, 10000, 20000, 30000]
    spending_levels = [params.annual_spending + d for d in spending_deltas if params.annual_spending + d > 0]
    spending_delta_labels = [f"{d//1000:+d}k" if d != 0 else "0" for d in spending_deltas if params.annual_spending + d > 0]

    with st.spinner(t("running_sensitivity", L)):
        sensitivity_grid, mortality_adjusted_grid = run_sensitivity_grid_fast(
            params, selected_returns, retirement_ages_sens, spending_levels, selected_mortality_table,
            num_simulations=200, calculate_mortality_adjusted_fn=calculate_mortality_adjusted_success
        )

    heatmap_col1, heatmap_col2 = st.columns(2)
    with heatmap_col1:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=sensitivity_grid, x=age_delta_labels, y=spending_delta_labels,
            colorscale=[[0, '#d73027'], [0.5, '#fee08b'], [0.85, '#1a9850'], [1, '#006837']],
            zmin=0, zmax=100, text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in sensitivity_grid],
            texttemplate="%{text}", textfont={"size": 11},
            hovertemplate="Retirement Age: %{customdata}<br>Spending: $%{meta:,}<br>Success Rate: %{z:.1f}%<extra></extra>",
            customdata=[[retirement_ages_sens[j] for j in range(len(retirement_ages_sens))] for _ in spending_levels],
            meta=[[spending_levels[i] for _ in retirement_ages_sens] for i in range(len(spending_levels))],
            colorbar=dict(title=t("success_rate", L), ticksuffix="%")
        ))
        fig_heatmap.update_layout(title=t("portfolio_survival_heatmap_title", L, age=params.end_age), xaxis_title=t("retirement_age_years", L), yaxis_title=t("spending_change", L), height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        figs.append(fig_heatmap)
        st.info(t("info_sensitivity_portfolio_survival", L, age=params.end_age))

    with heatmap_col2:
        fig_heatmap_mortality = go.Figure(data=go.Heatmap(
            z=mortality_adjusted_grid, x=age_delta_labels, y=spending_delta_labels,
            colorscale=[[0, '#d73027'], [0.5, '#fee08b'], [0.85, '#1a9850'], [1, '#006837']],
            zmin=0, zmax=100, text=[[f"{v:.0f}%" if not np.isnan(v) else "" for v in row] for row in mortality_adjusted_grid],
            texttemplate="%{text}", textfont={"size": 11},
            hovertemplate="Retirement Age: %{customdata}<br>Spending: $%{meta:,}<br>Mortality-Adjusted: %{z:.1f}%<extra></extra>",
            customdata=[[retirement_ages_sens[j] for j in range(len(retirement_ages_sens))] for _ in spending_levels],
            meta=[[spending_levels[i] for _ in retirement_ages_sens] for i in range(len(spending_levels))],
            colorbar=dict(title=t("success_rate", L), ticksuffix="%")
        ))
        fig_heatmap_mortality.update_layout(title=t("money_lasts_lifetime_title", L), xaxis_title=t("retirement_age_years", L), yaxis_title=t("spending_change", L), height=500)
        st.plotly_chart(fig_heatmap_mortality, use_container_width=True)
        figs.append(fig_heatmap_mortality)
        st.info(t("info_sensitivity_mortality_adjusted", L))

    with st.expander(t("historical_context", L), expanded=False):
        market_df = region_data['market_data_df']
        events_dict = region_data['historical_events']
        market_name = CONFIG[selected_region_key]['market_name']
        fig_historical = create_historical_events_chart(market_df, events_dict, market_name, L)
        st.plotly_chart(fig_historical, use_container_width=True)
        figs.append(fig_historical)

        st.subheader(t("key_historical_events", L))
        events_data = []
        for year, event in events_dict.items():
            if year in market_df.index:
                events_data.append({
                    t("year", L): year,
                    t("event", L): event['name'],
                    t("real_return", L): f"{market_df.loc[year, 'Real_Return']:.1%}",
                    t("description", L): event['description']
                })
        events_table = pd.DataFrame(events_data)
        st.dataframe(events_table, hide_index=True, use_container_width=True)

        st.subheader(t("stress_test_title", L), help=h("stress_test", L))
        st.write(t("stress_test_description", L))

        retirement_idx = params.retirement_age - params.current_age - 1
        if retirement_idx >= 0 and retirement_idx < len(results['median_simulation'].yearly_records):
            projected_retirement_balance = results['median_simulation'].yearly_records[retirement_idx].total_balance
        else:
            projected_retirement_balance = params.pretax_savings + params.posttax_savings
        
        years_in_retirement = params.end_age - params.retirement_age
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
        
        if stress_results:
            fig_stress = create_stress_test_chart(stress_results, events_dict, years_in_retirement, L)
            st.plotly_chart(fig_stress, use_container_width=True)
            figs.append(fig_stress)

            st.subheader(t("stress_test_results", L))
            stress_table_data = []
            for year, data in stress_results.items():
                event = events_dict[year]
                outcome = t("survived", L, balance=data['final_balance']) if data['survived'] else t("depleted", L, years=data['ruin_year'])
                stress_table_data.append({
                    t("year", L): year,
                    t("event", L): event['name'],
                    t("starting_balance", L): f"${projected_retirement_balance:,.0f}",
                    t("annual_spending", L): f"${params.annual_spending:,.0f}",
                    t("outcome", L): outcome
                })
            stress_table = pd.DataFrame(stress_table_data)
            st.dataframe(stress_table, hide_index=True, use_container_width=True)

            survivors = sum(1 for d in stress_results.values() if d['survived'])
            total = len(stress_results)
            if survivors == total:
                st.success(t("stress_all_survived", L, total=total))
            elif survivors == 0:
                st.error(t("stress_all_failed", L, total=total))
            else:
                st.warning(t("stress_partial", L, survivors=survivors, total=total))

    st.header(t("download_report", L))
    if st.button(t("generate_pdf", L)):
        with st.spinner(t("generating_pdf", L)):
            pdf_data = generate_pdf_report(st, st.session_state['params'], st.session_state['results'], figs)
            st.download_button(
                label=t("download_pdf", L),
                data=pdf_data,
                file_name="retirement_simulation_report.pdf",
                mime="application/pdf"
            )
