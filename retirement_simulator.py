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

# =============================================================================
# VECTORIZED IMPLEMENTATIONS FOR PERFORMANCE
# =============================================================================

def calculate_tax_us_vectorized(incomes: np.ndarray, brackets: List[Tuple[float, float]], standard_deduction: float) -> np.ndarray:
    """Vectorized US progressive tax calculation for array of incomes."""
    taxable = np.maximum(0, incomes - standard_deduction)
    tax = np.zeros_like(taxable)
    prev_limit = 0.0
    for bracket_limit, rate in brackets:
        if bracket_limit == float('inf'):
            bracket_limit = 1e12  # Large number for numpy
        bracket_income = np.clip(taxable - prev_limit, 0, bracket_limit - prev_limit)
        tax += bracket_income * rate
        prev_limit = bracket_limit
    return tax

def get_vectorized_tax_calculator(tax_region: str):
    """Returns a vectorized tax calculator function and its parameters."""
    tax_data = get_tax_data(tax_region)
    if tax_region == 'US':
        brackets, standard_deduction = tax_data
        return lambda incomes: calculate_tax_us_vectorized(incomes, brackets, standard_deduction)
    elif tax_region == 'BR':
        _, flat_rate = tax_data
        return lambda incomes: incomes * flat_rate
    else:
        raise ValueError(f"Unknown tax region: {tax_region}")

def generate_return_sequences_vectorized(returns: pd.Series, num_years: int, num_sequences: int, rng=None) -> np.ndarray:
    """Generate multiple return sequences at once using vectorized operations with block resampling."""
    if rng is None:
        rng = np.random.default_rng()

    years = returns.index.values
    return_values = returns.values
    n_years_data = len(years)

    # Pick random start indices for all sequences
    start_indices = rng.integers(0, n_years_data, size=num_sequences)
    sequences = np.empty((num_sequences, num_years), dtype=np.float64)

    for i in range(num_years):
        current_indices = start_indices + i
        # Block resampling: when we exceed data, pick new random start
        wrapped_mask = current_indices >= n_years_data
        if wrapped_mask.any():
            new_starts = rng.integers(0, n_years_data, size=wrapped_mask.sum())
            start_indices[wrapped_mask] = new_starts - i
            current_indices[wrapped_mask] = new_starts
        sequences[:, i] = return_values[current_indices % n_years_data]

    return sequences

def run_monte_carlo_fast(params: SimulationParams, returns: pd.Series, num_simulations: int = None, rng=None) -> Dict:
    """
    Fast vectorized Monte Carlo simulation optimized for sensitivity analysis.
    Returns only success_rate and survival_probability (no detailed yearly records).
    """
    if rng is None:
        rng = np.random.default_rng()

    num_sims = num_simulations if num_simulations else params.num_simulations
    num_years = params.end_age - params.current_age
    retirement_year_idx = params.retirement_age - params.current_age

    # Generate all return sequences at once
    return_sequences = generate_return_sequences_vectorized(returns, num_years, num_sims, rng)

    # Initialize arrays for all simulations
    pretax = np.full(num_sims, params.pretax_savings, dtype=np.float64)
    posttax = np.full(num_sims, params.posttax_savings, dtype=np.float64)
    salary = np.full(num_sims, params.salary, dtype=np.float64)

    success = np.ones(num_sims, dtype=bool)
    ruin_age = np.full(num_sims, params.end_age, dtype=np.int32)

    # Track solvency by age for survival probability
    solvent_by_age = np.ones((num_sims, num_years), dtype=bool)

    # Get vectorized tax calculator
    tax_calc = get_vectorized_tax_calculator(params.tax_region)

    for i in range(num_years):
        age = params.current_age + i
        market_return = return_sequences[:, i]

        if age < params.retirement_age:
            # Accumulation phase - fully vectorized
            pretax_contrib = salary * params.pretax_savings_rate
            posttax_contrib = salary * params.posttax_savings_rate
            match_eligible = salary * params.employer_match_cap
            employer_match = np.minimum(pretax_contrib, match_eligible) * params.employer_match_rate

            pretax += pretax_contrib + employer_match
            posttax += posttax_contrib
            pretax *= (1 + market_return)
            posttax *= (1 + market_return)
            salary *= (1 + params.salary_growth_rate)
        else:
            # Withdrawal phase - vectorized with iterative tax convergence
            spending_need = params.annual_spending
            remaining_need = np.full(num_sims, spending_need, dtype=np.float64)

            # Withdraw from post-tax first
            posttax_withdrawal = np.minimum(posttax, remaining_need)
            posttax -= posttax_withdrawal
            remaining_need -= posttax_withdrawal

            # Withdraw from pre-tax with tax gross-up (vectorized convergence)
            need_pretax_mask = (remaining_need > 0) & (pretax > 0)
            if need_pretax_mask.any():
                pretax_withdrawal = np.zeros(num_sims, dtype=np.float64)
                pretax_withdrawal[need_pretax_mask] = remaining_need[need_pretax_mask]

                # Iterative tax convergence (vectorized)
                prev_withdrawal = np.zeros(num_sims, dtype=np.float64)
                for _ in range(10):
                    tax = tax_calc(pretax_withdrawal)
                    needed_gross = remaining_need + tax
                    pretax_withdrawal = np.where(need_pretax_mask,
                                                  np.minimum(pretax, needed_gross),
                                                  pretax_withdrawal)
                    converged = np.abs(pretax_withdrawal - prev_withdrawal) < 0.01
                    if converged.all():
                        break
                    prev_withdrawal = pretax_withdrawal.copy()

                pretax -= pretax_withdrawal

            pretax *= (1 + market_return)
            posttax *= (1 + market_return)

            # Check for ruin
            total = pretax + posttax
            newly_ruined = (total <= 0) & success & (remaining_need > posttax_withdrawal)
            if newly_ruined.any():
                success[newly_ruined] = False
                ruin_age[newly_ruined] = age

            solvent_by_age[:, i] = (total > 0) | (age < params.retirement_age)

        pretax = np.maximum(pretax, 0)
        posttax = np.maximum(posttax, 0)

        # Track solvency for accumulation phase
        if age < params.retirement_age:
            solvent_by_age[:, i] = True

    # Calculate results
    success_rate = success.mean() * 100
    survival_prob = {params.current_age + i: solvent_by_age[:, i].mean() * 100 for i in range(num_years)}

    return {
        'success_rate': success_rate,
        'survival_probability': survival_prob,
    }

def run_sensitivity_grid_fast(params: SimulationParams, returns: pd.Series,
                               retirement_ages: List[int], spending_levels: List[float],
                               mortality_table: dict, num_simulations: int = 200,
                               calculate_mortality_adjusted_fn=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized sensitivity grid computation using vectorized Monte Carlo.

    This is significantly faster than running separate simulations for each grid point
    because it uses vectorized numpy operations instead of Python loops.
    """
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility across grid

    results_grid = np.zeros((len(spending_levels), len(retirement_ages)))
    mortality_adjusted_grid = np.zeros((len(spending_levels), len(retirement_ages)))

    for i, spending in enumerate(spending_levels):
        for j, ret_age in enumerate(retirement_ages):
            if spending <= 0 or ret_age <= params.current_age or ret_age >= params.end_age:
                results_grid[i, j] = np.nan
                mortality_adjusted_grid[i, j] = np.nan
                continue

            # Create modified params
            modified_params = SimulationParams(
                current_age=params.current_age, retirement_age=ret_age, end_age=params.end_age,
                pretax_savings=params.pretax_savings, posttax_savings=params.posttax_savings,
                salary=params.salary, salary_growth_rate=params.salary_growth_rate,
                pretax_savings_rate=params.pretax_savings_rate, posttax_savings_rate=params.posttax_savings_rate,
                employer_match_rate=params.employer_match_rate, employer_match_cap=params.employer_match_cap,
                annual_spending=spending, num_simulations=num_simulations, tax_region=params.tax_region
            )

            # Run fast Monte Carlo
            mc_results = run_monte_carlo_fast(modified_params, returns, num_simulations, rng)
            results_grid[i, j] = mc_results['success_rate']

            # Calculate mortality-adjusted success if function provided
            if calculate_mortality_adjusted_fn:
                mortality_adjusted_grid[i, j] = calculate_mortality_adjusted_fn(
                    mc_results, ret_age, params.end_age, mortality_table
                )

    return results_grid, mortality_adjusted_grid