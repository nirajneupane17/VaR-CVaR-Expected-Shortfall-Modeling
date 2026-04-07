# VaR, CVaR & Expected Shortfall Modeling

Quantitative risk framework for estimating and backtesting 
market risk measures across multi-asset portfolios — Historical 
VaR, Parametric VaR, Monte Carlo VaR, and Expected Shortfall.

---

## Overview

This project builds a comprehensive Value at Risk (VaR) and 
Expected Shortfall (ES) modeling framework from scratch using 
Python, covering three core methodologies, full backtesting 
workflows, and model governance documentation aligned with 
internal risk standards.

---

## Methodologies Covered

**Historical Simulation VaR**
- Rolling window estimation (252-day, 500-day)
- Non-parametric return distribution
- VaR at 95%, 99%, and 99.5% confidence levels

**Parametric VaR**
- Normal and Student-t distribution assumptions
- Variance-covariance matrix estimation
- Exponentially Weighted Moving Average (EWMA) volatility

**Monte Carlo VaR**
- Correlated asset return simulation
- Geometric Brownian Motion (GBM)
- 10,000 plus simulation paths

**Expected Shortfall (CVaR)**
- Conditional loss beyond VaR threshold
- Basel III/IV compliant ES at 97.5% confidence
- Comparison across all three methodologies

---

## Backtesting Framework

- Kupiec POF Test — tests exception frequency
- Christoffersen Independence Test — tests clustering 
  of exceptions
- Traffic Light Approach — Basel green/yellow/red zone 
  classification
- VaR exception analysis and breach reporting
- Model limitation documentation

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/NumPy-%230288D1.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-%234527A0.svg?style=for-the-badge&logo=pandas&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%231565C0.svg?style=for-the-badge&logo=scipy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23C62828.svg?style=for-the-badge&logo=Matplotlib&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%2300C853.svg?style=for-the-badge&logo=plotly&logoColor=white)

---

## Project Structure
```
VaR-CVaR-Expected-Shortfall-Modeling/
│
├── data/
│   └── market_data.csv
│
├── notebooks/
│   ├── 01_historical_var.ipynb
│   ├── 02_parametric_var.ipynb
│   ├── 03_monte_carlo_var.ipynb
│   ├── 04_expected_shortfall.ipynb
│   └── 05_backtesting.ipynb
│
├── src/
│   ├── var_models.py
│   ├── expected_shortfall.py
│   └── backtesting.py
│
├── results/
│   └── var_exception_report.csv
│
└── README.md
```

---

## Key Results

- Historical VaR (99%) backtesting exception rate within 
  Basel green zone across all tested portfolios
- Monte Carlo ES estimates converge within 0.5% of 
  Historical ES at 10,000 simulation paths
- EWMA volatility-adjusted VaR significantly reduces 
  exception clustering during high-volatility regimes

---

## Applications

- Multi-asset portfolio risk monitoring
- Regulatory capital calculation (Basel III/IV)
- Internal model governance and validation
- Daily risk reporting and threshold breach alerts

---

## References

- Basel Committee on Banking Supervision — FRTB Framework
- Jorion, P. — Value at Risk (3rd Edition)
- McNeil, Frey and Embrechts — Quantitative Risk Management
