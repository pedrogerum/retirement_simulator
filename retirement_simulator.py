"""
Retirement Planning Simulator - Core Simulation Engine
Uses historical S&P 500 REAL returns (inflation-adjusted) for Monte Carlo simulations.
All values are in TODAY'S DOLLARS - no inflation tracking needed.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

# 2024 US Federal Tax Brackets (Single Filer)
# Treated as constant since they historically adjust for inflation
TAX_BRACKETS = [
    (11600, 0.10),
    (47150, 0.12),
    (100525, 0.22),
    (191950, 0.24),
    (243725, 0.32),
    (609350, 0.35),
    (float('inf'), 0.37)
]
STANDARD_DEDUCTION = 14600


@dataclass
class SimulationParams:
    """Parameters for retirement simulation - all values in today's dollars"""
    # Personal
    current_age: int
    retirement_age: int
    end_age: int

    # Current savings (today's dollars)
    pretax_savings: float  # 401k, Traditional IRA
    posttax_savings: float  # Roth, Brokerage

    # Income & savings (today's dollars)
    salary: float
    salary_growth_rate: float  # REAL growth rate (above inflation), e.g., 0.01 for 1%
    pretax_savings_rate: float  # e.g., 0.15 for 15%
    posttax_savings_rate: float  # e.g., 0.05 for 5%
    employer_match_rate: float  # e.g., 0.50 for 50% match
    employer_match_cap: float  # e.g., 0.06 for up to 6% of salary

    # Retirement (today's dollars - stays constant since we use real returns)
    annual_spending: float

    # Simulation
    num_simulations: int
    tax_region: str = 'US'


@dataclass
class YearlyRecord:
    """Record for a single year in simulation - all values in today's dollars"""
    age: int
    pretax_balance: float
    posttax_balance: float
    total_balance: float
    contribution: float
    employer_match: float
    market_return: float  # Real return (inflation-adjusted)
    withdrawal: float
    tax_paid: float
    is_solvent: bool
    salary: float


@dataclass
class SimulationResult:
    """Result of a single simulation run - all values in today's dollars"""
    yearly_records: List[YearlyRecord]
    final_balance: float
    success: bool  # True if money lasted until end_age
    ruin_age: int  # Age when money ran out (or end_age if success)
    total_contributions: float
    total_growth: float


def calculate_tax(income: float) -> float:
    """Calculate federal income tax on pre-tax withdrawals"""
    taxable_income = max(0, income - STANDARD_DEDUCTION)
    tax = 0
    prev_bracket = 0

    for bracket_limit, rate in TAX_BRACKETS:
        if taxable_income <= 0:
            break
        taxable_in_bracket = min(taxable_income, bracket_limit - prev_bracket)
        tax += taxable_in_bracket * rate
        taxable_income -= taxable_in_bracket
        prev_bracket = bracket_limit

    return tax


# 2024 Brazil Federal Tax Brackets (Annual)
# Formula: (Income * Rate) - Deductible Portion
TAX_BRACKETS_BR = [
    (26963.20, 0.0, 0.0),
    (33919.80, 0.075, 2022.24),
    (45012.60, 0.15, 4566.23),
    (55976.16, 0.225, 7942.17),
    (float('inf'), 0.275, 10740.98)
]

def calculate_tax_br(income: float) -> float:
    """Calculate Brazilian federal income tax on pre-tax withdrawals."""
    return income * 0.10


def load_market_data(csv_path: str = None) -> pd.Series:
    """Load real (inflation-adjusted) S&P 500 returns from CSV"""
    if csv_path is None:
        # Prefer market_data.csv (has real returns)
        market_data_path = os.path.join(os.path.dirname(__file__), 'market_data.csv')
        if os.path.exists(market_data_path):
            csv_path = market_data_path
        else:
            csv_path = os.path.join(os.path.dirname(__file__), 'sp500_annual_returns.csv')

    df = pd.read_csv(csv_path)

    # Use Real_Return if available, otherwise fall back to nominal
    if 'Real_Return' in df.columns:
        returns = pd.Series(df['Real_Return'].values, index=df['Year'].values)
    elif 'Annual_Return' in df.columns:
        returns = pd.Series(df['Annual_Return'].values, index=df['Year'].values)
    else:
        returns = pd.Series(df['SP500_Return'].values, index=df['Year'].values)

    return returns


def get_return_sequence(returns: pd.Series, num_years: int, start_year: int = None) -> Tuple[np.ndarray, int]:
    """Get a sequence of real returns starting from a random or specified year"""
    years = returns.index.values

    if start_year is None:
        start_year = np.random.choice(years)

    sequence = []
    current_year = start_year

    for _ in range(num_years):
        if current_year in years:
            sequence.append(returns[current_year])
        else:
            # Wrap around if we run out of data
            current_year = years[0]
            sequence.append(returns[current_year])
        current_year += 1

    return np.array(sequence), start_year


def run_single_simulation(params: SimulationParams, returns: pd.Series,
                          start_year: int = None) -> SimulationResult:
    """
    Run a single Monte Carlo simulation using real (inflation-adjusted) returns.
    All values stay in today's dollars throughout.
    """

    num_years = params.end_age - params.current_age
    return_sequence, start_year = get_return_sequence(returns, num_years, start_year)

    pretax = params.pretax_savings
    posttax = params.posttax_savings
    salary = params.salary

    yearly_records = []
    total_contributions = 0
    success = True
    ruin_age = params.end_age

    for i, age in enumerate(range(params.current_age, params.end_age)):
        market_return = return_sequence[i]
        contribution = 0
        employer_match = 0
        withdrawal = 0
        tax_paid = 0

        # Accumulation phase
        if age < params.retirement_age:
            # Calculate contributions
            pretax_contrib = salary * params.pretax_savings_rate
            posttax_contrib = salary * params.posttax_savings_rate

            # Employer match (goes to pre-tax)
            match_eligible = salary * params.employer_match_cap
            employer_match = min(pretax_contrib, match_eligible) * params.employer_match_rate

            # Add contributions
            pretax += pretax_contrib + employer_match
            posttax += posttax_contrib

            contribution = pretax_contrib + posttax_contrib
            total_contributions += contribution + employer_match

            # Apply real market returns
            pretax *= (1 + market_return)
            posttax *= (1 + market_return)

            # Grow salary by real growth rate
            salary *= (1 + params.salary_growth_rate)

        # Retirement phase
        else:
            # Spending is CONSTANT in today's dollars (no inflation adjustment needed)
            spending_need = params.annual_spending

            # Withdraw (post-tax first, then pre-tax)
            remaining_need = spending_need

            # Draw from post-tax (Roth) first - tax free
            posttax_withdrawal = min(posttax, remaining_need)
            posttax -= posttax_withdrawal
            remaining_need -= posttax_withdrawal

            # Draw from pre-tax (401k) - taxable
            if remaining_need > 0 and pretax > 0:
                # Iterate to find gross withdrawal that covers need + taxes
                pretax_withdrawal = remaining_need
                for _ in range(5):  # Iterate to converge
                    if params.tax_region == 'BR':
                        tax = calculate_tax_br(pretax_withdrawal)
                    else: # Default to US
                        tax = calculate_tax(pretax_withdrawal)
                    
                    needed_gross = remaining_need + tax
                    pretax_withdrawal = min(pretax, needed_gross)

                if params.tax_region == 'BR':
                    tax_paid = calculate_tax_br(pretax_withdrawal)
                else: # Default to US
                    tax_paid = calculate_tax(pretax_withdrawal)
                pretax -= pretax_withdrawal
                remaining_need -= (pretax_withdrawal - tax_paid)

            withdrawal = spending_need

            # Apply real market returns to remaining balance
            pretax *= (1 + market_return)
            posttax *= (1 + market_return)

            # Check for ruin
            if pretax <= 0 and posttax <= 0 and remaining_need > 0:
                pretax = 0
                posttax = 0
                if success:  # First time hitting ruin
                    success = False
                    ruin_age = age

        # Ensure non-negative
        pretax = max(0, pretax)
        posttax = max(0, posttax)

        yearly_records.append(YearlyRecord(
            age=age,
            pretax_balance=pretax,
            posttax_balance=posttax,
            total_balance=pretax + posttax,
            contribution=contribution,
            employer_match=employer_match,
            market_return=market_return,
            withdrawal=withdrawal,
            tax_paid=tax_paid,
            is_solvent=(pretax + posttax) > 0 or age < params.retirement_age,
            salary=salary
        ))

    final_balance = pretax + posttax
    total_growth = final_balance + sum(r.withdrawal for r in yearly_records) - total_contributions - params.pretax_savings - params.posttax_savings

    return SimulationResult(
        yearly_records=yearly_records,
        final_balance=final_balance,
        success=success,
        ruin_age=ruin_age,
        total_contributions=total_contributions,
        total_growth=total_growth
    )


def run_monte_carlo(params: SimulationParams, returns: pd.Series = None) -> Dict:
    """Run multiple simulations and aggregate results"""

    if returns is None:
        returns = load_market_data()

    results = []
    for _ in range(params.num_simulations):
        result = run_single_simulation(params, returns)
        results.append(result)

    # Aggregate metrics - all values already in today's dollars
    final_balances = [r.final_balance for r in results]
    success_count = sum(1 for r in results if r.success)
    ruin_ages = [r.ruin_age for r in results if not r.success]

    # Calculate percentiles for each age
    balances_by_age = {age: [] for age in range(params.current_age, params.end_age)}
    solvent_by_age = {age: 0 for age in range(params.current_age, params.end_age)}
    pretax_by_age = {age: [] for age in range(params.current_age, params.end_age)}
    posttax_by_age = {age: [] for age in range(params.current_age, params.end_age)}
    withdrawals_by_age = {age: [] for age in range(params.current_age, params.end_age)}

    for result in results:
        for record in result.yearly_records:
            balances_by_age[record.age].append(record.total_balance)
            pretax_by_age[record.age].append(record.pretax_balance)
            posttax_by_age[record.age].append(record.posttax_balance)
            withdrawals_by_age[record.age].append(record.withdrawal)
            if record.is_solvent:
                solvent_by_age[record.age] += 1

    # Calculate percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    balance_percentiles = {}
    for age in range(params.current_age, params.end_age):
        balance_percentiles[age] = {
            p: np.percentile(balances_by_age[age], p) for p in percentiles
        }

    # Survival probability
    survival_prob = {
        age: solvent_by_age[age] / params.num_simulations * 100
        for age in range(params.current_age, params.end_age)
    }

    # Contribution vs growth at retirement
    retirement_idx = params.retirement_age - params.current_age - 1
    if retirement_idx >= 0:
        contributions_at_retirement = [
            sum(r.yearly_records[i].contribution + r.yearly_records[i].employer_match
                for i in range(retirement_idx + 1))
            for r in results
        ]
        balances_at_retirement = [
            r.yearly_records[retirement_idx].total_balance for r in results
        ]
        median_contrib_retirement = np.median(contributions_at_retirement)
        median_balance_retirement = np.median(balances_at_retirement)
        median_growth_retirement = median_balance_retirement - median_contrib_retirement - params.pretax_savings - params.posttax_savings
    else:
        median_contrib_retirement = 0
        median_balance_retirement = params.pretax_savings + params.posttax_savings
        median_growth_retirement = 0

    # Median values by age
    median_pretax_by_age = {age: np.median(pretax_by_age[age]) for age in range(params.current_age, params.end_age)}
    median_posttax_by_age = {age: np.median(posttax_by_age[age]) for age in range(params.current_age, params.end_age)}
    median_withdrawals_by_age = {age: np.median(withdrawals_by_age[age]) for age in range(params.current_age, params.end_age)}

    # Find median simulation for cash flow table
    median_final = np.median(final_balances)
    median_idx = np.argmin([abs(r.final_balance - median_final) for r in results])
    median_simulation = results[median_idx]

    return {
        'success_rate': success_count / params.num_simulations * 100,
        'median_final_balance': np.median(final_balances),
        'percentile_5_final': np.percentile(final_balances, 5),
        'percentile_95_final': np.percentile(final_balances, 95),
        'mean_final_balance': np.mean(final_balances),
        'std_final_balance': np.std(final_balances),
        'probability_of_ruin': (1 - success_count / params.num_simulations) * 100,
        'average_shortfall_years': np.mean([params.end_age - age for age in ruin_ages]) if ruin_ages else 0,
        'balance_percentiles': balance_percentiles,
        'survival_probability': survival_prob,
        'final_balances': final_balances,
        'results': results,
        'median_simulation': median_simulation,
        'contribution_at_retirement': median_contrib_retirement + params.pretax_savings + params.posttax_savings,
        'growth_at_retirement': median_growth_retirement,
        'median_pretax_by_age': median_pretax_by_age,
        'median_posttax_by_age': median_posttax_by_age,
        'median_withdrawals_by_age': median_withdrawals_by_age,
    }


def run_sensitivity_analysis(params: SimulationParams, returns: pd.Series = None) -> Dict:
    """Run simulations at different spending levels"""
    if returns is None:
        returns = load_market_data()

    spending_levels = [
        params.annual_spending - 20000,
        params.annual_spending - 10000,
        params.annual_spending,
        params.annual_spending + 10000,
        params.annual_spending + 20000,
    ]

    results = {}
    for spending in spending_levels:
        if spending <= 0:
            continue
        
        modified_params = SimulationParams(
            current_age=params.current_age,
            retirement_age=params.retirement_age,
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
            num_simulations=params.num_simulations,
            tax_region=params.tax_region
        )
        mc_results = run_monte_carlo(modified_params, returns)
        results[spending] = {
            'spending': modified_params.annual_spending,
            'success_rate': mc_results['success_rate'],
            'median_final': mc_results['median_final_balance']
        }

    return results


def run_retirement_age_comparison(params: SimulationParams, returns: pd.Series = None) -> Dict:
    """Compare different retirement ages"""
    if returns is None:
        returns = load_market_data()

    retirement_ages = [
        params.retirement_age - 2,
        params.retirement_age - 1,
        params.retirement_age,
        params.retirement_age + 1,
        params.retirement_age + 2,
    ]

    results = {}
    for ret_age in retirement_ages:
        if ret_age <= params.current_age or ret_age >= params.end_age:
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
            annual_spending=params.annual_spending,
            num_simulations=params.num_simulations,
            tax_region=params.tax_region
        )
        mc_results = run_monte_carlo(modified_params, returns)
        results[ret_age] = {
            'success_rate': mc_results['success_rate'],
            'median_final': mc_results['median_final_balance'],
            'percentile_5': mc_results['percentile_5_final'],
            'percentile_95': mc_results['percentile_95_final']
        }

    return results


if __name__ == "__main__":
    # Example usage
    params = SimulationParams(
        current_age=35,
        retirement_age=65,
        end_age=95,
        pretax_savings=150000,
        posttax_savings=50000,
        salary=100000,
        salary_growth_rate=0.01,  # 1% real growth (above inflation)
        pretax_savings_rate=0.15,
        posttax_savings_rate=0.05,
        employer_match_rate=0.50,
        employer_match_cap=0.06,
        annual_spending=60000,
        num_simulations=1000,
        tax_region='US'
    )

    print("Running Monte Carlo simulation (using real returns)...")
    print("All values in today's dollars")
    results = run_monte_carlo(params)

    print(f"\nResults:")
    print(f"  Success Rate: {results['success_rate']:.1f}%")
    print(f"  Median Final Balance: ${results['median_final_balance']:,.0f}")
    print(f"  5th Percentile: ${results['percentile_5_final']:,.0f}")
    print(f"  95th Percentile: ${results['percentile_95_final']:,.0f}")
    print(f"  Probability of Ruin: {results['probability_of_ruin']:.1f}%")
    if results['average_shortfall_years'] > 0:
        print(f"  Average Shortfall: {results['average_shortfall_years']:.1f} years")
