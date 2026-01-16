# app/services/temp_service/utils.py
from typing import List, Optional, Dict
import pandas as pd
import numpy as np


def safe_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return None
    if a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
        return None
    return float(a[mask].corr(b[mask]))


def series_to_pylist(s: pd.Series) -> List[Optional[float]]:
    return [None if (pd.isna(x)) else float(x) for x in s.tolist()]


def _align_dicts_to_arrays(d1: Dict[str, Optional[float]], d2: Dict[str, Optional[float]]):
    keys = sorted(set(d1.keys()) & set(d2.keys()))
    if not keys:
        return np.array([], dtype=float), np.array([], dtype=float), []
    a = []
    b = []
    for k in keys:
        v1 = d1.get(k, None)
        v2 = d2.get(k, None)
        a.append(np.nan if v1 is None else float(v1))
        b.append(np.nan if v2 is None else float(v2))
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float), keys


def safe_corr_vec(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return None
    xs = x[mask]
    ys = y[mask]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return None
    try:
        r = float(np.corrcoef(xs, ys)[0, 1])
        if not np.isfinite(r):
            return None
        return r
    except Exception:
        return None
