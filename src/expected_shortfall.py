"""
expected_shortfall.py
=====================
Expected Shortfall (CVaR) estimation — Historical, Parametric,
and rolling window implementations.
Basel III/IV standard: ES at 97.5% confidence level.

Author : Niraj Neupane | github.com/nirajneupane17
Project: VaR-CVaR-Expected-Shortfall-Modeling
"""

import numpy as np
import pandas as pd
from scipy import stats


def historical_es(returns: pd.Series,
                  confidence_levels: list = [0.95, 0.975, 0.99]) -> pd.DataFrame:
    """
    Historical Expected Shortfall (CVaR).
    ES = average loss in the tail beyond the VaR threshold.

    Basel III/IV standard uses ES at 97.5% confidence level.

    Returns
    -------
    pd.DataFrame with VaR, ES, ES/VaR ratio, tail observation count
    """
    rows = []
    for cl in confidence_levels:
        var_thresh = np.percentile(returns, (1-cl)*100)
        tail = returns[returns <= var_thresh]
        var_val = abs(var_thresh)
        es_val  = abs(tail.mean()) if len(tail) > 0 else var_val
        rows.append({
            'confidence_level': f'{int(cl*100)}%',
            'VaR':          round(var_val, 6),
            'ES (CVaR)':    round(es_val, 6),
            'ES/VaR ratio': round(es_val/var_val, 3),
            'Tail obs':     len(tail)
        })
    return pd.DataFrame(rows).set_index('confidence_level')


def rolling_es(returns: pd.Series,
               window: int = 252,
               confidence_level: float = 0.975) -> pd.DataFrame:
    """
    Rolling window Expected Shortfall.

    Parameters
    ----------
    window           : rolling window in trading days (default 252)
    confidence_level : Basel III/IV default is 0.975 (97.5%)

    Returns
    -------
    pd.DataFrame with rolling VaR and ES time series
    """
    var_list, es_list = [], []
    for i in range(window, len(returns)+1):
        w = returns.iloc[i-window:i]
        var_t = abs(np.percentile(w, (1-confidence_level)*100))
        tail  = w[w <= -var_t]
        es_t  = abs(tail.mean()) if len(tail) > 0 else var_t
        var_list.append(var_t)
        es_list.append(es_t)
    idx = returns.index[window-1:]
    cl_label = int(confidence_level*100)
    return pd.DataFrame({
        f'VaR_{cl_label}pct': var_list,
        f'ES_{cl_label}pct':  es_list
    }, index=idx)


def parametric_es_normal(returns: pd.Series,
                          confidence_levels: list = [0.95, 0.975, 0.99]) -> pd.DataFrame:
    """
    Parametric Expected Shortfall under Normal distribution.

    ES_normal = sigma * phi(z) / (1-cl) - mu
    where phi is the standard normal PDF and z = norm.ppf(1-cl)
    """
    mu, sig = returns.mean(), returns.std()
    rows = []
    for cl in confidence_levels:
        z = stats.norm.ppf(1-cl)
        var_val = abs(mu + z*sig)
        es_val  = sig * stats.norm.pdf(z) / (1-cl) - mu
        rows.append({
            'confidence_level': f'{int(cl*100)}%',
            'VaR':          round(var_val, 6),
            'ES (CVaR)':    round(es_val,  6),
            'ES/VaR ratio': round(es_val/var_val, 3)
        })
    return pd.DataFrame(rows).set_index('confidence_level')
