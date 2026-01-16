#!/usr/bin/env python
import pandas as pd
from bcb import sgs
import traceback

print("Script starting...")

try:
    # ============================================================
    # SELIC (CDI) and INFLATION (IPCA)
    # ============================================================
    print("=" * 60)
    print("Downloading Brazil historical data (CDI and IPCA)...")
    print("=" * 60)

    # Fetch monthly IPCA (inflation)
    ipca = sgs.get({'ipca': 433}, start='1980-01-01')
    ipca['ipca'] = ipca['ipca'] / 100  # Convert from percentage to decimal

    # Fetch daily CDI in chunks of 9 years
    today = pd.to_datetime('today')
    start_year = 1980
    cdi_chunks = []
    while start_year <= today.year:
        start_date = f'{start_year}-01-01'
        end_year = start_year + 8
        end_date = f'{end_year}-12-31'
        if pd.to_datetime(end_date) > today:
            end_date = today.strftime('%Y-%m-%d')
        
        print(f"Fetching CDI data from {start_date} to {end_date}...")
        chunk = sgs.get({'cdi': 12}, start=start_date, end=end_date)
        cdi_chunks.append(chunk)
        
        if pd.to_datetime(end_date).year >= today.year:
            break
        start_year = end_year + 1

    cdi = pd.concat(cdi_chunks)
    cdi['cdi'] = cdi['cdi'] / 100 # Convert from percentage to decimal

    # ============================================================
    # CALCULATE ANNUAL RETURNS
    # ============================================================
    print("\n" + "=" * 60)
    print("Calculating annual returns...")
    print("=" * 60)

    # Calculate annual inflation from monthly IPCA
    # The annual inflation is the product of (1 + monthly inflation) for the year
    annual_inflation = (1 + ipca['ipca']).resample('YE').prod() - 1
    annual_inflation = annual_inflation.to_frame(name='inflation')

    # Calculate annual CDI from daily CDI
    # The annual CDI is the product of (1 + daily CDI) for the year
    annual_cdi = (1 + cdi['cdi']).resample('YE').prod() - 1
    annual_cdi = annual_cdi.to_frame(name='cdi')

    # Merge into a single dataframe
    df = pd.merge(annual_cdi, annual_inflation, left_index=True, right_index=True)
    df['year'] = df.index.year
    df.set_index('year', inplace=True)

    # Calculate real return
    df['real_return'] = (1 + df['cdi']) / (1 + df['inflation']) - 1

    # Rename columns to match the desired format
    df.rename(columns={'cdi': 'CDI_Return', 'inflation': 'Inflation_Rate', 'real_return': 'Real_Return'}, inplace=True)
    df.index.name = 'Year'


    # ============================================================
    # SAVE FILES
    # ============================================================
    print("\n" + "=" * 60)
    print("Saving files...")
    print("=" * 60)

    df.to_csv('market_data_br.csv')

    print(f"\nFiles saved:")
    print(f"  - market_data_br.csv (CDI returns + inflation)")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nYears covered: {df.index.min()} to {df.index.max()} ({len(df)} years)")

    print(f"\nCDI Nominal Returns:")
    print(f"  Mean:   {df['CDI_Return'].mean():.2%}")
    print(f"  Median: {df['CDI_Return'].median():.2%}")
    print(f"  Std:    {df['CDI_Return'].std():.2%}")
    print(f"  Min:    {df['CDI_Return'].min():.2%} ({df['CDI_Return'].idxmin()})")
    print(f"  Max:    {df['CDI_Return'].max():.2%} ({df['CDI_Return'].idxmax()})")

    print(f"\nInflation (IPCA):")
    print(f"  Mean:   {df['Inflation_Rate'].mean():.2%}")
    print(f"  Median: {df['Inflation_Rate'].median():.2%}")
    print(f"  Min:    {df['Inflation_Rate'].min():.2%} ({df['Inflation_Rate'].idxmin()})")
    print(f"  Max:    {df['Inflation_Rate'].max():.2%} ({df['Inflation_Rate'].idxmax()})")

    print(f"\nCDI Real Returns (inflation-adjusted):")
    print(f"  Mean:   {df['Real_Return'].mean():.2%}")
    print(f"  Median: {df['Real_Return'].median():.2%}")
    print(f"\nSample data (last 10 years):")
    print(df.tail(10).to_string())

except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()

print("Script finished.")
