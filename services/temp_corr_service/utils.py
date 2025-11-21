# app/services/temp_service/utils.py
from typing import List, Optional
import pandas as pd


def safe_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return None
    if a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
        return None
    return float(a[mask].corr(b[mask]))


def series_to_pylist(s: pd.Series) -> List[Optional[float]]:
    return [None if (pd.isna(x)) else float(x) for x in s.tolist()]
