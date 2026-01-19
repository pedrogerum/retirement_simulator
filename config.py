"""
Configuration for the Retirement Simulator
This file centralizes the settings for different regions.
"""

# In a real-world scenario, you might load this from a YAML or JSON file.
CONFIG = {
    'US': {
        'name': 'US (S&P 500)',
        'market_data': 'data/market_data.csv',
        'historical_events': 'data/historical_events_us.csv',
        'mortality_table': 'data/mortality_us.csv',
        'tax_brackets': 'data/us_tax_brackets.csv',
        'tax_function': 'calculate_tax_us',
        'market_name': 'S&P 500'
    },
    'BR': {
        'name': 'Brazil (CDI)',
        'market_data': 'data/market_data_br.csv',
        'historical_events': 'data/historical_events_br.csv',
        'mortality_table': 'data/mortality_br.csv',
        'tax_brackets': 'data/br_tax_brackets.csv',
        'tax_function': 'calculate_tax_br',
        'market_name': 'CDI (Brazil)'
    }
}
