"""
backtesting.py
==============
VaR backtesting framework — Kupiec POF test, Christoffersen
Independence test, and Basel Traffic Light classification.

Author : Niraj Neupane | github.com/nirajneupane17
Project: VaR-CVaR-Expected-Shortfall-Modeling
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_exceptions(actual_returns: pd.Series,
                        var_estimates: pd.Series) -> pd.Series:
    """
    Identify VaR exceptions — days where realised loss exceeds VaR.

    Parameters
    ----------
    actual_returns : realised portfolio returns (pd.Series)
    var_estimates  : VaR estimates expressed as positive losses (pd.Series)

    Returns
    -------
    pd.Series of binary exception indicators (1 = breach, 0 = no breach)
    """
    aligned = actual_returns.align(var_estimates, join='inner')
    return (aligned[0] < -aligned[1]).astype(int)


def kupiec_pof_test(n_obs: int, n_exc: int,
                    confidence_level: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) test.

    H0: True exception rate equals (1 - confidence_level)
    LR statistic follows chi-squared distribution with df=1.
    Reject H0 (model fails) if LR > 3.841 (5% significance).

    Parameters
    ----------
    n_obs            : total number of observations
    n_exc            : number of VaR exceptions
    confidence_level : VaR confidence level (e.g. 0.99 for 99% VaR)

    Returns
    -------
    dict with LR statistic, p-value, critical value, pass/fail result
    """
    p     = 1 - confidence_level
    p_hat = np.clip(n_exc / n_obs, 1e-10, 1-1e-10)

    lr = -2 * (
        n_exc     * np.log(p     / p_hat) +
        (n_obs - n_exc) * np.log((1-p) / (1-p_hat))
    )
    critical = stats.chi2.ppf(0.95, df=1)
    p_value  = 1 - stats.chi2.cdf(lr, df=1)

    return {
        'test':             'Kupiec POF',
        'n_obs':            n_obs,
        'n_exceptions':     n_exc,
        'exception_rate':   round(p_hat, 4),
        'expected_rate':    round(p, 4),
        'LR_statistic':     round(lr, 4),
        'critical_value':   round(critical, 4),
        'p_value':          round(p_value, 4),
        'reject_H0':        lr > critical,
        'model_valid':      lr <= critical
    }


def christoffersen_test(exceptions: pd.Series) -> dict:
    """
    Christoffersen Independence Test.

    Tests whether VaR exceptions cluster (occur on consecutive days).
    Clustering indicates the model underestimates tail risk during stress.

    H0: Exceptions are independently distributed (no clustering).
    Reject H0 if LR_ind > 3.841 (5% significance).

    Returns
    -------
    dict with LR statistic, p-value, clustering indicator
    """
    exc = exceptions.values
    n00 = ((exc[:-1]==0) & (exc[1:]==0)).sum()
    n01 = ((exc[:-1]==0) & (exc[1:]==1)).sum()
    n10 = ((exc[:-1]==1) & (exc[1:]==0)).sum()
    n11 = ((exc[:-1]==1) & (exc[1:]==1)).sum()

    p01 = np.clip(n01/(n00+n01) if (n00+n01)>0 else 1e-10, 1e-10, 1-1e-10)
    p11 = np.clip(n11/(n10+n11) if (n10+n11)>0 else 1e-10, 1e-10, 1-1e-10)
    p   = np.clip((n01+n11)/len(exc), 1e-10, 1-1e-10)

    lr_ind = -2 * (
        (n00+n10)*np.log(1-p) + (n01+n11)*np.log(p) -
        n00*np.log(1-p01) - n01*np.log(p01) -
        n10*np.log(1-p11) - n11*np.log(p11)
    )
    critical = stats.chi2.ppf(0.95, df=1)
    p_value  = 1 - stats.chi2.cdf(lr_ind, df=1)

    return {
        'test':           'Christoffersen Independence',
        'LR_statistic':   round(lr_ind, 4),
        'critical_value': round(critical, 4),
        'p_value':        round(p_value, 4),
        'clustering':     lr_ind > critical,
        'independent':    lr_ind <= critical
    }


def basel_traffic_light(n_exc: int, n_obs: int = 250) -> dict:
    """
    Basel Traffic Light classification for 99% VaR backtest.

    Standard Basel window: 250 trading days (~1 year)
    Green  : 0-4  exceptions — model accepted
    Yellow : 5-9  exceptions — investigate
    Red    : 10+  exceptions — model rejected

    Returns
    -------
    dict with zone classification, capital add-on, and required action
    """
    if n_exc <= 4:
        zone, addon, action = 'GREEN',  0,    'Model accepted — no action required'
    elif n_exc <= 9:
        zone, addon, action = 'YELLOW', n_exc-4, 'Investigate model assumptions'
    else:
        zone, addon, action = 'RED',    4,    'Model rejected — revise immediately'

    return {
        'zone':          zone,
        'n_exceptions':  n_exc,
        'capital_addon': addon,
        'action':        action
    }


def full_backtest(actual_returns: pd.Series,
                  var_estimates: pd.Series,
                  confidence_level: float = 0.99) -> dict:
    """
    Run complete backtesting suite.

    Returns dict with exceptions summary, Kupiec, Christoffersen,
    and Basel Traffic Light results.
    """
    exc   = compute_exceptions(actual_returns, var_estimates)
    n_obs = len(exc)
    n_exc = int(exc.sum())
    return {
        'summary':        {'n_obs': n_obs, 'n_exceptions': n_exc,
                           'exception_rate': round(n_exc/n_obs, 4)},
        'kupiec':         kupiec_pof_test(n_obs, n_exc, confidence_level),
        'christoffersen': christoffersen_test(exc),
        'basel':          basel_traffic_light(n_exc)
    }
