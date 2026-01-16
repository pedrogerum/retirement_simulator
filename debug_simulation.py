"""Debug script to trace simulation values"""
import pandas as pd
import numpy as np
from retirement_simulator import SimulationParams, run_single_simulation, load_sp500_returns

# User's parameters
params = SimulationParams(
    current_age=35,
    retirement_age=65,
    end_age=95,
    pretax_savings=275800,
    posttax_savings=12827,
    salary=150000,
    salary_growth_rate=0.03,
    pretax_savings_rate=0.14,
    posttax_savings_rate=0.0,  # Assuming 0 since not mentioned
    employer_match_rate=1.0,   # 100% match
    employer_match_cap=0.095,  # up to 9.5%
    annual_spending=80000,
    inflation_rate=0.025,
    num_simulations=1000
)

# Load returns
returns = load_sp500_returns()
print("S&P 500 Returns loaded:")
print(f"  Years: {returns.index.min()} to {returns.index.max()}")
print(f"  Mean return: {returns.mean():.2%}")
print(f"  Median return: {returns.median():.2%}")
print(f"  Min return: {returns.min():.2%} ({returns.idxmin()})")
print(f"  Max return: {returns.max():.2%} ({returns.idxmax()})")
print()

# Run a single simulation with a BAD starting year (1929 - Great Depression)
print("=" * 60)
print("SIMULATION STARTING 1929 (Great Depression)")
print("=" * 60)
result_1929 = run_single_simulation(params, returns, start_year=1929)

print(f"\nKey milestones:")
for record in result_1929.yearly_records:
    if record.age in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 94]:
        phase = "ACCUM" if record.age < 65 else "RETIRE"
        print(f"  Age {record.age} [{phase}]: ${record.total_balance:,.0f} (return: {record.market_return:.1%})")

print(f"\nFinal balance: ${result_1929.final_balance:,.0f}")
print(f"Success: {result_1929.success}")
print()

# Run a single simulation with a GOOD starting year (1980 - Bull market)
print("=" * 60)
print("SIMULATION STARTING 1980 (Bull market)")
print("=" * 60)
result_1980 = run_single_simulation(params, returns, start_year=1980)

print(f"\nKey milestones:")
for record in result_1980.yearly_records:
    if record.age in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 94]:
        phase = "ACCUM" if record.age < 65 else "RETIRE"
        print(f"  Age {record.age} [{phase}]: ${record.total_balance:,.0f} (return: {record.market_return:.1%})")

print(f"\nFinal balance: ${result_1980.final_balance:,.0f}")
print(f"Success: {result_1980.success}")
print()

# Run a single simulation with 2000 (dot-com crash)
print("=" * 60)
print("SIMULATION STARTING 2000 (Dot-com crash)")
print("=" * 60)
result_2000 = run_single_simulation(params, returns, start_year=2000)

print(f"\nKey milestones:")
for record in result_2000.yearly_records:
    if record.age in [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 94]:
        phase = "ACCUM" if record.age < 65 else "RETIRE"
        print(f"  Age {record.age} [{phase}]: ${record.total_balance:,.0f} (return: {record.market_return:.1%})")

print(f"\nFinal balance: ${result_2000.final_balance:,.0f}")
print(f"Success: {result_2000.success}")
print()

# Check what actual sequence of returns we get for a 60-year simulation starting 1929
print("=" * 60)
print("RETURN SEQUENCE CHECK (60 years from 1929)")
print("=" * 60)
years_needed = 60
start = 1929
sequence_years = []
for i in range(years_needed):
    year = start + i
    if year > returns.index.max():
        year = returns.index.min() + (year - returns.index.max() - 1)
    sequence_years.append(year)

print(f"Years used: {sequence_years[:10]}...{sequence_years[-10:]}")
print(f"Note: Wraps from {returns.index.max()} back to {returns.index.min()}")

# Quick sanity check - what's the compound return over 30 years?
print()
print("=" * 60)
print("30-YEAR COMPOUND RETURNS FROM DIFFERENT START YEARS")
print("=" * 60)
for start_year in [1929, 1950, 1970, 1980, 1990, 2000]:
    compound = 1.0
    for i in range(30):
        year = start_year + i
        if year in returns.index:
            compound *= (1 + returns[year])
        else:
            # Wrap
            wrap_year = returns.index.min() + (year - returns.index.max() - 1)
            if wrap_year in returns.index:
                compound *= (1 + returns[wrap_year])
    annualized = compound ** (1/30) - 1
    print(f"  Starting {start_year}: {compound:.1f}x total ({annualized:.1%} annualized)")
