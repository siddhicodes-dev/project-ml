
# Political Uncertainty & Corporate Investment

## Overview
This project investigates the **causal impact of political uncertainty on corporate investment decisions** using panel econometrics and modern causal machine learning methods (Double ML, IPW). It analyzes quarterly financial data from U.S. public companies and the Economic Policy Uncertainty (EPU) Index to quantify how uncertainty shocks affect investment behavior.

---

## Goals
1. Determine whether political uncertainty causally reduces corporate investment.
2. Quantify the economic magnitude of investment reduction during high uncertainty periods.
3. Validate robustness across multiple methods (panel econometrics and ML).

---

## Data
- **EPU Index**: Monthly data (1985–2025) from [policyuncertainty.com](https://www.policyuncertainty.com/)  
- **Financial Data**: US quarterly company data from SimFin (Balance sheets, Cash flows, Income statements)  

**Key Variables**
| Variable | Description |
|----------|-------------|
| Investment Ratio | Net investing cash flow / Total Assets |
| Cash Flow Ratio | Operating cash flow / Total Assets |
| Leverage | Total Liabilities / Total Assets |
| Size | Log(Total Assets) |
| Profitability | Net Income / Total Assets |

---

## Installation

```bash
git clone https://github.com/siddhicodes-dev/project-ml.git
cd project-ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# project-ml
