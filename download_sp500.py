import yfinance as yf
import pandas as pd

print("=" * 60)
print("Downloading S&P 500 historical data...")
print("=" * 60)

# Download S&P 500 historical data (^GSPC is the Yahoo Finance ticker)
sp500 = yf.download("^GSPC", start="1927-01-01", end="2025-01-01")

# Flatten multi-index columns if present
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

# Use 'Close' if 'Adj Close' not available (yfinance version differences)
price_col = 'Adj Close' if 'Adj Close' in sp500.columns else 'Close'

# Calculate daily returns
sp500['Daily_Return'] = sp500[price_col].pct_change()

# Calculate annual returns (using year-end prices)
annual_data = sp500[price_col].resample('YE').last()
annual_returns = annual_data.pct_change().dropna()

# Create annual returns dataframe
annual_df = pd.DataFrame({
    'Year': annual_returns.index.year,
    'SP500_Return': annual_returns.values
})

print(f"\nS&P 500 data from {sp500.index[0].date()} to {sp500.index[-1].date()}")
print(f"  Years: {len(annual_df)}")
print(f"  Mean return: {annual_df['SP500_Return'].mean():.2%}")

# ============================================================
# HISTORICAL INFLATION DATA (CPI)
# ============================================================
print("\n" + "=" * 60)
print("Adding historical inflation data (CPI)...")
print("=" * 60)

# Historical US CPI data (annual averages, Bureau of Labor Statistics CPI-U)
cpi_historical = {
    1913: 9.9, 1914: 10.0, 1915: 10.1, 1916: 10.9, 1917: 12.8, 1918: 15.1, 1919: 17.3, 1920: 20.0,
    1921: 17.9, 1922: 16.8, 1923: 17.1, 1924: 17.1, 1925: 17.5, 1926: 17.7, 1927: 17.4, 1928: 17.2,
    1929: 17.2, 1930: 16.7, 1931: 15.2, 1932: 13.6, 1933: 12.9, 1934: 13.4, 1935: 13.7, 1936: 13.9,
    1937: 14.4, 1938: 14.1, 1939: 13.9, 1940: 14.0, 1941: 14.7, 1942: 16.3, 1943: 17.3, 1944: 17.6,
    1945: 18.0, 1946: 19.5, 1947: 22.3, 1948: 24.1, 1949: 23.8, 1950: 24.1, 1951: 26.0, 1952: 26.6,
    1953: 26.8, 1954: 26.9, 1955: 26.8, 1956: 27.2, 1957: 28.1, 1958: 28.9, 1959: 29.2, 1960: 29.6,
    1961: 29.9, 1962: 30.3, 1963: 30.6, 1964: 31.0, 1965: 31.5, 1966: 32.5, 1967: 33.4, 1968: 34.8,
    1969: 36.7, 1970: 38.8, 1971: 40.5, 1972: 41.8, 1973: 44.4, 1974: 49.3, 1975: 53.8, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9, 1982: 96.5, 1983: 99.6, 1984: 103.9,
    1985: 107.6, 1986: 109.6, 1987: 113.6, 1988: 118.3, 1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3,
    1993: 144.5, 1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6, 2000: 172.2,
    2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6, 2007: 207.3, 2008: 215.3,
    2009: 214.5, 2010: 218.1, 2011: 224.9, 2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0,
    2017: 245.1, 2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.7, 2024: 314.5,
}

# Create inflation dataframe
cpi_df = pd.DataFrame({'Year': list(cpi_historical.keys()), 'CPI': list(cpi_historical.values())})
cpi_df = cpi_df.sort_values('Year')
cpi_df['Inflation_Rate'] = cpi_df['CPI'].pct_change()
cpi_df = cpi_df.dropna()

print(f"  Inflation data from {int(cpi_df['Year'].min())} to {int(cpi_df['Year'].max())}")
print(f"  Mean inflation: {cpi_df['Inflation_Rate'].mean():.2%}")

# ============================================================
# MERGE DATA
# ============================================================
print("\n" + "=" * 60)
print("Merging S&P 500 returns with inflation...")
print("=" * 60)

# Merge S&P 500 returns with inflation
merged_df = pd.merge(annual_df, cpi_df[['Year', 'Inflation_Rate']], on='Year', how='left')

# Fill any missing inflation values with historical average
avg_inflation = cpi_df['Inflation_Rate'].mean()
missing_years = merged_df[merged_df['Inflation_Rate'].isna()]['Year'].tolist()
if missing_years:
    print(f"  Missing inflation for years: {missing_years}")
    print(f"  Filling with average inflation: {avg_inflation:.2%}")
merged_df['Inflation_Rate'] = merged_df['Inflation_Rate'].fillna(avg_inflation)

# Calculate real return (nominal return adjusted for inflation)
# Real Return = (1 + Nominal) / (1 + Inflation) - 1
merged_df['Real_Return'] = (1 + merged_df['SP500_Return']) / (1 + merged_df['Inflation_Rate']) - 1

# ============================================================
# SAVE FILES
# ============================================================
print("\n" + "=" * 60)
print("Saving files...")
print("=" * 60)

# Save merged data (main file for simulation)
merged_df.to_csv('market_data.csv', index=False)

# Also save daily data for reference
sp500.to_csv('sp500_daily.csv')

# Keep backward compatibility - also save as sp500_annual_returns.csv
# (in case other code references it)
backward_compat_df = merged_df.rename(columns={'SP500_Return': 'Annual_Return'})
backward_compat_df.to_csv('sp500_annual_returns.csv', index=False)

print(f"\nFiles saved:")
print(f"  - market_data.csv (S&P 500 returns + inflation)")
print(f"  - sp500_annual_returns.csv (backward compatible)")
print(f"  - sp500_daily.csv (daily prices)")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nYears covered: {int(merged_df['Year'].min())} to {int(merged_df['Year'].max())} ({len(merged_df)} years)")

print(f"\nS&P 500 Nominal Returns:")
print(f"  Mean:   {merged_df['SP500_Return'].mean():.2%}")
print(f"  Median: {merged_df['SP500_Return'].median():.2%}")
print(f"  Std:    {merged_df['SP500_Return'].std():.2%}")
print(f"  Min:    {merged_df['SP500_Return'].min():.2%} ({int(merged_df.loc[merged_df['SP500_Return'].idxmin(), 'Year'])})")
print(f"  Max:    {merged_df['SP500_Return'].max():.2%} ({int(merged_df.loc[merged_df['SP500_Return'].idxmax(), 'Year'])})")

print(f"\nInflation:")
print(f"  Mean:   {merged_df['Inflation_Rate'].mean():.2%}")
print(f"  Median: {merged_df['Inflation_Rate'].median():.2%}")
print(f"  Min:    {merged_df['Inflation_Rate'].min():.2%} ({int(merged_df.loc[merged_df['Inflation_Rate'].idxmin(), 'Year'])})")
print(f"  Max:    {merged_df['Inflation_Rate'].max():.2%} ({int(merged_df.loc[merged_df['Inflation_Rate'].idxmax(), 'Year'])})")

print(f"\nS&P 500 Real Returns (inflation-adjusted):")
print(f"  Mean:   {merged_df['Real_Return'].mean():.2%}")
print(f"  Median: {merged_df['Real_Return'].median():.2%}")

print(f"\nSample data (last 10 years):")
print(merged_df.tail(10).to_string(index=False))
