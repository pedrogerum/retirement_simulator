"""
Retirement Planning Simulator - Core Simulation Engine
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import os
from config import CONFIG

def load_us_tax_brackets() -> Tuple[List[Tuple[float, float]], float]:
    csv_path = os.path.join(os.path.dirname(__file__), CONFIG['US']['tax_brackets'])
    df = pd.read_csv(csv_path)
    brackets = []
    standard_deduction = 0
    for _, row in df.iterrows():
        limit = float('inf') if row['bracket_limit'] == 'inf' else float(row['bracket_limit'])
        rate = float(row['rate'])
        brackets.append((limit, rate))
        if pd.notna(row['standard_deduction']) and row['standard_deduction'] != '':
            standard_deduction = float(row['standard_deduction'])
    return brackets, standard_deduction

def load_br_tax_brackets() -> Tuple[List[Tuple[float, float, float]], float]:
    csv_path = os.path.join(os.path.dirname(__file__), CONFIG['BR']['tax_brackets'])
    df = pd.read_csv(csv_path)
    brackets = []
    flat_rate = 0.10
    for _, row in df.iterrows():
        limit = float('inf') if row['bracket_limit'] == 'inf' else float(row['bracket_limit'])
        rate = float(row['rate'])
        deduction = float(row['deduction'])
        brackets.append((limit, rate, deduction))
        if pd.notna(row['flat_rate']) and row['flat_rate'] != '':
            flat_rate = float(row['flat_rate'])
    return brackets, flat_rate

def get_tax_data(tax_region: str) -> Tuple:
    if tax_region == 'US':
        return load_us_tax_brackets()
    elif tax_region == 'BR':
        return load_br_tax_brackets()
    else:
        raise ValueError(f"Unknown tax region: {tax_region}")

@dataclass
class SimulationParams:
    current_age: int
    retirement_age: int
    end_age: int
    pretax_savings: float
    posttax_savings: float
    salary: float
    salary_growth_rate: float
    pretax_savings_rate: float
    posttax_savings_rate: float
    employer_match_rate: float
    employer_match_cap: float
    annual_spending: float
    num_simulations: int
    tax_region: str = 'US'

@dataclass
class YearlyRecord:
    age: int
    pretax_balance: float
    posttax_balance: float
    total_balance: float
    contribution: float
    employer_match: float
    market_return: float
    withdrawal: float
    tax_paid: float
    is_solvent: bool
    salary: float

@dataclass
class SimulationResult:
    yearly_records: List[YearlyRecord]
    final_balance: float
    success: bool
    ruin_age: int
    total_contributions: float
    total_growth: float
    balance_at_death: float

def calculate_tax_us(income: float, brackets: List[Tuple[float, float]], standard_deduction: float) -> float:
    taxable_income = max(0, income - standard_deduction)
    tax = 0
    prev_bracket_limit = 0
    for bracket_limit, rate in brackets:
        if taxable_income <= 0: break
        taxable_in_bracket = min(taxable_income, bracket_limit - prev_bracket_limit)
        tax += taxable_in_bracket * rate
        taxable_income -= taxable_in_bracket
        prev_bracket_limit = bracket_limit
    return tax

def calculate_tax_br(income: float, flat_rate: float) -> float:
    return income * flat_rate

def get_tax_calculator(tax_region: str) -> Callable[[float], float]:
    tax_data = get_tax_data(tax_region)
    if tax_region == 'US':
        brackets, standard_deduction = tax_data
        return lambda income: calculate_tax_us(income, brackets, standard_deduction)
    elif tax_region == 'BR':
        _, flat_rate = tax_data
        return lambda income: calculate_tax_br(income, flat_rate)
    else:
        raise ValueError(f"Unknown tax region: {tax_region}")

def load_market_data(region_key: str) -> pd.Series:
    csv_path = os.path.join(os.path.dirname(__file__), CONFIG[region_key]['market_data'])
    df = pd.read_csv(csv_path)
    if 'Real_Return' in df.columns:
        return pd.Series(df['Real_Return'].values, index=df['Year'].values)
    elif 'Annual_Return' in df.columns:
        return pd.Series(df['Annual_Return'].values, index=df['Year'].values)
    else:
        return pd.Series(df['SP500_Return'].values, index=df['Year'].values)

def get_return_sequence(returns: pd.Series, num_years: int, start_year: int = None) -> Tuple[np.ndarray, int]:
    years = returns.index.values
    if start_year is None:
        start_year = np.random.choice(years)
    sequence = []
    current_year = start_year
    for _ in range(num_years):
        if current_year in years:
            sequence.append(returns[current_year])
        else:
            # Block resampling: pick random new start point, continue sequentially
            current_year = np.random.choice(years)
            sequence.append(returns[current_year])
        current_year += 1
    return np.array(sequence), start_year

def _get_age_of_death(current_age, end_age, mortality_table):
    for age in range(current_age, end_age + 1):
        death_prob = mortality_table.get(age, 1.0)
        if np.random.random() < death_prob:
            return age
    return end_age

def run_single_simulation(params: SimulationParams, returns: pd.Series, mortality_table: dict, start_year: int = None) -> SimulationResult:
    num_years = params.end_age - params.current_age
    return_sequence, _ = get_return_sequence(returns, num_years, start_year)
    pretax, posttax, salary = params.pretax_savings, params.posttax_savings, params.salary
    yearly_records, total_contributions, success, ruin_age = [], 0, True, params.end_age
    tax_calculator = get_tax_calculator(params.tax_region)

    for i, age in enumerate(range(params.current_age, params.end_age)):
        market_return = return_sequence[i]
        contribution, employer_match, withdrawal, tax_paid = 0, 0, 0, 0
        if age < params.retirement_age:
            pretax_contrib = salary * params.pretax_savings_rate
            posttax_contrib = salary * params.posttax_savings_rate
            match_eligible = salary * params.employer_match_cap
            employer_match = min(pretax_contrib, match_eligible) * params.employer_match_rate
            pretax += pretax_contrib + employer_match
            posttax += posttax_contrib
            contribution = pretax_contrib + posttax_contrib
            total_contributions += contribution + employer_match
            pretax *= (1 + market_return)
            posttax *= (1 + market_return)
            salary *= (1 + params.salary_growth_rate)
        else:
            spending_need = params.annual_spending
            remaining_need = spending_need
            posttax_withdrawal = min(posttax, remaining_need)
            posttax -= posttax_withdrawal
            remaining_need -= posttax_withdrawal
            if remaining_need > 0 and pretax > 0:
                pretax_withdrawal = remaining_need
                prev_withdrawal = 0
                for _ in range(10):  # More iterations with convergence check
                    tax = tax_calculator(pretax_withdrawal)
                    needed_gross = remaining_need + tax
                    pretax_withdrawal = min(pretax, needed_gross)
                    if abs(pretax_withdrawal - prev_withdrawal) < 0.01:
                        break
                    prev_withdrawal = pretax_withdrawal
                tax_paid = tax_calculator(pretax_withdrawal)
                pretax -= pretax_withdrawal
            withdrawal = spending_need
            pretax *= (1 + market_return)
            posttax *= (1 + market_return)
            if pretax <= 0 and posttax <= 0 and remaining_need > 0:
                pretax, posttax = 0, 0
                if success:
                    success, ruin_age = False, age
        
        pretax, posttax = max(0, pretax), max(0, posttax)
        yearly_records.append(YearlyRecord(
            age=age, pretax_balance=pretax, posttax_balance=posttax, total_balance=pretax + posttax,
            contribution=contribution, employer_match=employer_match, market_return=market_return,
            withdrawal=withdrawal, tax_paid=tax_paid, is_solvent=(pretax + posttax) > 0 or age < params.retirement_age, salary=salary
        ))

    final_balance = pretax + posttax
    total_growth = final_balance + sum(r.withdrawal for r in yearly_records) - total_contributions - params.pretax_savings - params.posttax_savings
    age_of_death = _get_age_of_death(params.current_age, params.end_age, mortality_table)
    balance_at_death = next((r.total_balance for r in yearly_records if r.age == age_of_death), final_balance)

    return SimulationResult(
        yearly_records=yearly_records, final_balance=final_balance, success=success,
        ruin_age=ruin_age, total_contributions=total_contributions,
        total_growth=total_growth, balance_at_death=balance_at_death
    )

def run_monte_carlo(params: SimulationParams, returns: pd.Series, mortality_table: dict) -> Dict:

    results = [run_single_simulation(params, returns, mortality_table) for _ in range(params.num_simulations)]



    final_balances = [r.final_balance for r in results]

    balances_at_death = [r.balance_at_death for r in results]

    success_count = sum(1 for r in results if r.success)

    ruin_ages = [r.ruin_age for r in results if not r.success]



    balances_by_age = {age: [r.yearly_records[age - params.current_age].total_balance for r in results if len(r.yearly_records) > (age - params.current_age)] for age in range(params.current_age, params.end_age)}

    

    solvent_by_age = {age: sum(1 for r in results if len(r.yearly_records) > (age - params.current_age) and r.yearly_records[age - params.current_age].is_solvent) for age in range(params.current_age, params.end_age)}



    percentiles = [5, 10, 25, 50, 75, 90, 95]

    balance_percentiles = {age: {p: np.percentile(balances_by_age[age], p) for p in percentiles if balances_by_age[age]} for age, balances in balances_by_age.items()}

    survival_prob = {age: count / params.num_simulations * 100 for age, count in solvent_by_age.items()}



    retirement_idx = params.retirement_age - params.current_age - 1

    if retirement_idx >= 0 and any(len(r.yearly_records) > retirement_idx for r in results):

        median_balance_retirement = np.median([r.yearly_records[retirement_idx].total_balance for r in results if len(r.yearly_records) > retirement_idx])

        median_contrib_retirement = np.median([sum(rec.contribution + rec.employer_match for rec in r.yearly_records[:retirement_idx + 1]) for r in results if len(r.yearly_records) > retirement_idx])

        median_growth_retirement = median_balance_retirement - median_contrib_retirement - params.pretax_savings - params.posttax_savings

    else:

        median_contrib_retirement, median_balance_retirement, median_growth_retirement = 0, params.pretax_savings + params.posttax_savings, 0

    

    median_final = np.median(final_balances)

    median_idx = np.argmin([abs(r.final_balance - median_final) for r in results])



    return {

        'success_rate': success_count / params.num_simulations * 100,

        'median_final_balance': median_final,

        'percentile_5_final': np.percentile(final_balances, 5),

        'percentile_95_final': np.percentile(final_balances, 95),

        'probability_of_ruin': (1 - success_count / params.num_simulations) * 100,

        'average_shortfall_years': np.mean([params.end_age - age for age in ruin_ages]) if ruin_ages else 0,

        'balance_percentiles': balance_percentiles,

        'survival_probability': survival_prob,

        'final_balances': final_balances,

        'balances_at_death': balances_at_death,

        'median_simulation': results[median_idx],

        'contribution_at_retirement': median_contrib_retirement + params.pretax_savings + params.posttax_savings,

        'growth_at_retirement': median_growth_retirement,

    }