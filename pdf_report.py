import pandas as pd
from fpdf import FPDF
import time
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Retirement Simulation Report', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(st, params, results, figs):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # --- Assumptions ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Simulation Assumptions', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)

    assumptions = {
        "Market Data Source": params.tax_region,
        "Current Age": params.current_age,
        "Retirement Age": params.retirement_age,
        "End Age": params.end_age,
        "Pre-tax Savings": f"${params.pretax_savings:,.0f}",
        "Post-tax Savings": f"${params.posttax_savings:,.0f}",
        "Current Annual Salary": f"${params.salary:,.0f}",
        "Real Salary Growth Rate": f"{params.salary_growth_rate:.2%}",
        "Pre-tax Savings Rate": f"{params.pretax_savings_rate:.2%}",
        "Post-tax Savings Rate": f"{params.posttax_savings_rate:.2%}",
        "Employer Match Rate": f"{params.employer_match_rate:.2%}",
        "Employer Match Cap": f"{params.employer_match_cap:.2%}",
        "Desired Annual Spending in Retirement": f"${params.annual_spending:,.0f}",
        "Number of Monte Carlo Simulations": params.num_simulations,
    }

    for key, value in assumptions.items():
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(90, 8, f"{key}:", 0, 0, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(90, 8, str(value), 0, 1, 'L')
    pdf.ln(5)

    # --- Methodology ---
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Methodology', 0, 1, 'L')
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Interest and Inflation Methodology', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5,
        "The simulation uses historical real returns, which are already adjusted for inflation. "
        "This means all financial values throughout the simulation are in 'today's dollars', "
        "and there is no need to model inflation separately.\n\n"
        f"The data source for the {params.tax_region} market is '{'market_data_br.csv' if params.tax_region == 'BR' else 'market_data.csv'}'. "
        "The simulation randomly samples historical yearly returns from this dataset."
    )
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Tax Calculation', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    if params.tax_region == 'US':
        pdf.multi_cell(0, 5, 
            "US federal income tax is calculated on pre-tax withdrawals during retirement. "
            "The calculation uses the 2024 tax brackets for a single filer, which are assumed to adjust with inflation. "
            "A standard deduction of $14,600 is applied."
        )
    else:
        pdf.multi_cell(0, 5, 
            "Brazilian tax is calculated as a fixed 10% on all pre-tax withdrawals during retirement. "
            "No deductions are applied."
        )
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Retirement Withdrawal Strategy', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, 
        "The withdrawal strategy prioritizes post-tax (e.g., Roth) accounts first to meet annual spending needs. "
        "Once post-tax accounts are depleted, withdrawals are made from pre-tax (e.g., 401k) accounts. "
        "Taxes are calculated and paid on the amounts withdrawn from pre-tax accounts."
    )
    pdf.ln(10)

    # --- Results ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Simulation Results', 0, 1, 'L')
    
    # Key Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Key Metrics', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    metrics = {
        "Success Rate": f"{results['success_rate']:.1f}%",
        "Median Final Balance": f"${results['median_final_balance']:,.0f}",
        "5th Percentile Final Balance": f"${results['percentile_5_final']:,.0f}",
        "95th Percentile Final Balance": f"${results['percentile_95_final']:,.0f}",
    }
    for key, value in metrics.items():
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(90, 8, f"{key}:", 0, 0, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(90, 8, str(value), 0, 1, 'L')
    pdf.ln(5)

    # Add charts
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Charts', 0, 1, 'L')

    chart_paths = []
    charts_generated = False

    for i, fig in enumerate(figs):
        chart_title = fig.layout.title.text if fig.layout.title.text else f"Chart {i+1}"
        st.text(f"Generating chart: {chart_title}...")
        path = f"chart_{i}.png"

        try:
            start_time = time.time()
            fig.write_image(path)
            end_time = time.time()

            time_taken = end_time - start_time
            print(f"Time taken for '{chart_title}': {time_taken:.2f} seconds")
            st.text(f"'{chart_title}' generated in {time_taken:.2f} seconds.")

            chart_paths.append(path)
            charts_generated = True
        except Exception as e:
            st.text(f"Could not generate chart '{chart_title}': Image export not available")
            print(f"Chart generation error for '{chart_title}': {e}")
            # Only show the warning once
            if i == 0:
                pdf.set_font('Arial', 'I', 10)
                pdf.multi_cell(0, 5,
                    "Note: Charts could not be included in this PDF. "
                    "Image export requires additional dependencies (kaleido) that are not available in this environment. "
                    "Please view the charts in the interactive Streamlit app."
                )
                pdf.ln(5)
            break

    if charts_generated:
        st.text("Charts generated. Creating PDF...")
        for path in chart_paths:
            if os.path.exists(path):
                pdf.image(path, w=180)
                pdf.ln(2)
                # Clean up temporary file
                try:
                    os.remove(path)
                except:
                    pass
    else:
        st.text("Creating PDF without charts...")

    return bytes(pdf.output(dest='S'))
