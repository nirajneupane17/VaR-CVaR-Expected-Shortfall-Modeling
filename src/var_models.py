"""
var_models.py
=============
Core VaR estimation — Historical, Parametric (Normal, Student-t,
EWMA), and Monte Carlo methodologies.

Author : Niraj Neupane | github.com/nirajneupane17
Project: VaR-CVaR-Expected-Shortfall-Modeling
"""

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series,
                   window: int = 252,
                   confidence_levels: list = [0.95, 0.99, 0.995]) -> pd.DataFrame:
    """
    Rolling Historical Simulation VaR.

    Parameters
    ----------
    returns           : daily portfolio returns (pd.Series)
    window            : rolling window in trading days (default 252)
    confidence_levels : list of confidence levels e.g. [0.95, 0.99]

    Returns
    -------
    pd.DataFrame with rolling VaR estimates expressed as positive loss
    """
    results = {}
    for cl in confidence_levels:
        col = [abs(np.percentile(returns.iloc[i-window:i], (1-cl)*100))
               for i in range(window, len(returns)+1)]
        results[f'VaR_{int(cl*100)}pct'] = col
    return pd.DataFrame(results, index=returns.index[window-1:])


def normal_var(returns: pd.Series,
               confidence_levels: list = [0.95, 0.99, 0.995]) -> dict:
    """
    Parametric VaR under Normal distribution assumption.

    Returns dict {label: VaR_value}
    """
    mu, sig = returns.mean(), returns.std()
    return {
        f'Normal_VaR_{int(cl*100)}pct': abs(mu + stats.norm.ppf(1-cl) * sig)
        for cl in confidence_levels
    }


def student_t_var(returns: pd.Series,
                  confidence_levels: list = [0.95, 0.99, 0.995]) -> dict:
    """
    Parametric VaR using fitted Student-t distribution.
    Captures fat tails present in financial return data.

    Returns dict with VaR estimates and fitted degrees of freedom.
    """
    df_t, loc_t, scale_t = stats.t.fit(returns)
    result = {'fitted_df': round(df_t, 4)}
    for cl in confidence_levels:
        q = stats.t.ppf(1-cl, df=df_t, loc=loc_t, scale=scale_t)
        result[f'StudentT_VaR_{int(cl*100)}pct'] = abs(q)
    return result


def ewma_var(returns: pd.Series,
             lam: float = 0.94,
             confidence_levels: list = [0.95, 0.99, 0.995]) -> dict:
    """
    EWMA volatility-adjusted VaR (RiskMetrics methodology).
    Lambda=0.94 is the RiskMetrics standard for daily data.

    Returns dict with EWMA VaR estimates and current EWMA volatility.
    """
    ewma_vol = np.sqrt(returns.ewm(span=(2/(1-lam)-1)).var().iloc[-1])
    mu = returns.mean()
    result = {'ewma_vol': round(ewma_vol, 6)}
    for cl in confidence_levels:
        result[f'EWMA_VaR_{int(cl*100)}pct'] = abs(mu + stats.norm.ppf(1-cl) * ewma_vol)
    return result


def monte_carlo_var(returns: pd.DataFrame,
                    weights: np.ndarray,
                    n_sims: int = 10000,
                    confidence_levels: list = [0.95, 0.99, 0.995],
                    seed: int = 42) -> pd.DataFrame:
    """
    Monte Carlo VaR using correlated multivariate normal simulation.

    Parameters
    ----------
    returns    : multi-asset daily returns (pd.DataFrame)
    weights    : portfolio weights array (must sum to 1)
    n_sims     : number of Monte Carlo paths (default 10,000)
    seed       : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with MC VaR, MC ES, and ES/VaR ratio
    """
    np.random.seed(seed)
    sims = np.random.multivariate_normal(returns.mean().values,
                                          returns.cov().values, n_sims)
    port = sims @ weights
    rows = []
    for cl in confidence_levels:
        var_val = abs(np.percentile(port, (1-cl)*100))
        tail    = port[port <= -var_val]
        es_val  = abs(tail.mean()) if len(tail) > 0 else var_val
        rows.append({
            'confidence_level': f'{int(cl*100)}%',
            'MC_VaR':    round(var_val, 6),
            'MC_ES':     round(es_val,  6),
            'ES_VaR_ratio': round(es_val/var_val, 3)
        })
    return pd.DataFrame(rows).set_index('confidence_level')


def var_summary_table(returns: pd.Series,
                       confidence_levels: list = [0.90, 0.95, 0.99, 0.995]) -> pd.DataFrame:
    """
    Full VaR summary: daily, 10-day (Basel), and annualised.

    Returns pd.DataFrame with all scaling horizons.
    """
    rows = []
    for cl in confidence_levels:
        v = abs(np.percentile(returns, (1-cl)*100))
        rows.append({
            'confidence': f'{int(cl*100)}%',
            'daily_VaR':   round(v, 4),
            '10day_VaR':   round(v * np.sqrt(10),  4),
            'annual_VaR':  round(v * np.sqrt(252), 4)
        })
    return pd.DataFrame(rows).set_index('confidence')
