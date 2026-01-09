import numpy as np
from typing import Tuple, List, Optional

def _prepare_series(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two series by removing positions where either is NaN.
    Returns: (a_clean, b_clean) as 1-D numpy arrays of same length.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape[0] != b.shape[0]:
        L = min(a.shape[0], b.shape[0])
        a = a[:L]
        b = b[:L]
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]

def dtw_distance(a: np.ndarray,
                 b: np.ndarray,
                 window: Optional[int] = None,
                 dist_func = None,
                 return_path: bool = False) -> Tuple[float, Optional[List[Tuple[int,int]]]]:
    """
    Compute DTW distance between two 1-D arrays a and b.

    Args:
      a, b: 1-D numeric arrays (will be aligned by dropping NaNs).
      window: Sakoe-Chiba window size (in indices). If None -> no constraint.
              If int, a[i] can only match b[j] if |i-j| <= window.
      dist_func: function(x,y) -> nonnegative distance (default: abs(x-y)).
      return_path: if True, also return the optimal alignment path as list of (i,j).

    Returns:
      (distance, path_or_None)
    """
    a, b = _prepare_series(a, b)
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        if n == 0 and m == 0:
            return 0.0, [] if return_path else 0.0
        return float('inf'), [] if return_path else float('inf')

    if dist_func is None:
        def dist_func(x, y): return abs(x - y)

    if window is None:
        w = max(n, m)  # effectively no constraint
    else:
        w = int(window)
        w = max(w, abs(n - m))  # ensure feasible

    # initialize cost matrix with +inf
    INF = np.inf
    D = np.full((n+1, m+1), INF, dtype=float)
    D[0,0] = 0.0

    # fill DP (1..n, 1..m)
    for i in range(1, n+1):
        jmin = max(1, i - w)
        jmax = min(m, i + w)
        # vectorized local cost for this row if desired; but loop is fine
        for j in range(jmin, jmax+1):
            cost = dist_func(a[i-1], b[j-1])
            # transitions: (i-1,j), (i,j-1), (i-1,j-1)
            D[i,j] = cost + min(D[i-1, j],    # insertion
                                 D[i, j-1],  # deletion
                                 D[i-1, j-1])  # match

    dtw_dist = float(D[n, m])

    if not return_path:
        return dtw_dist, None

    path = []
    i, j = n, m
    while (i > 0) or (j > 0):
        path.append((i-1, j-1))
        choices = []
        if i > 0 and j > 0:
            choices.append((D[i-1, j-1], i-1, j-1))
        if i > 0:
            choices.append((D[i-1, j], i-1, j))
        if j > 0:
            choices.append((D[i, j-1], i, j-1))
        vals = [(c, ii, jj) for (c, ii, jj) in choices]
        cmin, i_prev, j_prev = min(vals, key=lambda x: x[0])
        i, j = i_prev, j_prev
    path.reverse()
    return dtw_dist, path

import numpy as np
from typing import Literal

def dtw_similarity(dtw_dist: float,
                   a: np.ndarray,
                   b: np.ndarray,
                   scale: Literal['length_std','inv','linear']='length_std') -> float:
    """
    Convert DTW distance to similarity in [0,1].

    Options:
      - 'length_std' (default): normalize by (L * pooled_std) then similarity = exp(-dist / norm_factor)
      - 'inv' : similarity = 1 / (1 + dist / norm_factor)
      - 'linear' : similarity = max(0, 1 - dist / norm_factor)

    """
    try:
        if not np.isfinite(dtw_dist):
            return 0.0
    except Exception:
        return 0.0

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    L = max(1, min(a.size, b.size))

    std_a = float(np.nanstd(a)) if a.size > 0 else 0.0
    std_b = float(np.nanstd(b)) if b.size > 0 else 0.0

    if std_a > 0 and std_b > 0:
        pooled_std = float(np.sqrt((std_a**2 + std_b**2) / 2.0))
    else:
        pooled_std = max(std_a, std_b, 1e-8)

    norm_factor = float(L * pooled_std)
    if not np.isfinite(norm_factor) or norm_factor <= 0:
        norm_factor = 1.0

    dist = float(dtw_dist)
    if dist < 0:
        dist = 0.0

    scale = str(scale).lower()
    if scale == 'length_std':
        sim = float(np.exp(- dist / norm_factor))
    elif scale == 'inv':
        sim = 1.0 / (1.0 + (dist / norm_factor))
    elif scale == 'linear':
        sim = 1.0 - min(dist / norm_factor, 1.0)
    else:
        raise ValueError(f"Unknown scale '{scale}'. Choose one of 'length_std','inv','linear'.")

    if not np.isfinite(sim):
        return 0.0
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return float(sim)
