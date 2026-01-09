import re
import logging
from typing import Dict, List, Optional, Iterable
import pandas as pd
import numpy as np


class DataValidator:

    @staticmethod
    def is_valid_pack_cell_df(
            df: pd.DataFrame,
            expected: int = 102,
            logger: logging.Logger = None
    ) -> bool:
        log = logger or logging.getLogger(__name__)

        if 'pack_code' not in df.columns:
            log.error("输入 DataFrame 中缺少列 'pack_code'，无法校验")
            return False

        pack_code = df['pack_code'].dropna().unique().tolist()

        if len(df) != expected:
                log.error(
                    "当前pack码 %s -> 电芯码数量 %d -> 期望 %d -> 数据有误，不做处理",
                    pack_code, len(df), expected
                )
                return False

        return True

    @staticmethod
    def _normalize_required_steps(required_steps: Optional[Iterable[int]]) -> set:
        if required_steps is None:
            return set(range(1, 10))
        return set(int(x) for x in required_steps)

    @staticmethod
    def is_valid_pack_result_df(
            df: pd.DataFrame,
            required_steps: Optional[Iterable[int]] = None,
            logger: Optional[logging.Logger] = None
    ) -> bool:

        log = logger or logging.getLogger(__name__)
        req_steps = DataValidator._normalize_required_steps(required_steps)

        if 'pack_code' not in df.columns:
            log.error("输入 DataFrame 缺少列 'pack_code'，无法校验")
            return False
        if 'step_id' not in df.columns:
            log.error("输入 DataFrame 缺少列 'step_id'，无法校验")
            return False

        pack_code = df['pack_code'].dropna().unique().tolist()
        step_nums = pd.to_numeric(df['step_id'], errors='coerce').dropna().astype(int).unique()
        step_set = set(step_nums)
        missing_steps = sorted(list(req_steps - step_set))
        if missing_steps:
            log.error(
                "当前pack码 %s -> 缺失必要电测工步 %s -> 数据有误不做处理",
                pack_code, missing_steps
            )
            return False

        return True


class PackFrameBuilder:

    def __init__(self,
                 volt_prefix: str = "bms_cellvolt",
                 cell_index_col: str = "cell_index",
                 step_id_col: str = "step_id",
                 logger: Optional[logging.Logger] = None):
        self.volt_prefix = volt_prefix
        self.cell_index_col = cell_index_col
        self.step_id_col = step_id_col
        self.logger = logger or logging.getLogger(__name__)
        self._idx_re = re.compile(r"(\d+)")

    def build_frames(self,
                     cell_df: pd.DataFrame,
                     result_df: pd.DataFrame) -> pd.DataFrame:

        if self.step_id_col not in result_df.columns:
            raise KeyError(f"Result DataFrame missing required column '{self.step_id_col}'")

        volt_cols = [c for c in result_df.columns if c and c.startswith(self.volt_prefix)]

        use_cols = [self.step_id_col] + volt_cols
        result_df = result_df[[c for c in use_cols if c in result_df.columns]]
        df = self.pivot_steps_to_columns_and_merge(cell_df, result_df)

        return df

    def pivot_steps_to_columns_and_merge(self,
                                         df_cell: pd.DataFrame,
                                         df_result: pd.DataFrame) -> pd.DataFrame:

        volt_cols = [c for c in df_result.columns if c and c.startswith(self.volt_prefix)]
        id_cols = [c for c in df_result.columns if c not in volt_cols]

        if not volt_cols:
            self.logger.debug("No voltage columns found with prefix '%s' in df_result; returning df_cell as-is", self.volt_prefix)
            return df_cell

        df2_long = df_result.melt(id_vars=id_cols, value_vars=volt_cols,
                                  var_name='bms_col', value_name='volt')

        def _extract_index(s):
            if s is None:
                return np.nan
            m = self._idx_re.search(s)
            return int(m.group(1)) if m else np.nan

        df2_long['cell_index'] = df2_long['bms_col'].astype(str).apply(_extract_index)
        missing_idx_mask = df2_long['cell_index'].isna()
        if missing_idx_mask.any():
            self.logger.warning("Some bms_col names did not contain an integer index; dropping %d rows", missing_idx_mask.sum())
            df2_long = df2_long[~missing_idx_mask].copy()

        if len(id_cols) == 1:
            idcol = id_cols[0]
            df2_long['step_col'] = df2_long[idcol].astype(str).apply(lambda x: f"step_{x}_volt")
        else:
            df2_long['step_col'] = df2_long[id_cols].astype(str).agg('_'.join, axis=1).apply(lambda x: f"step_{x}_volt")

        pivot = df2_long.pivot_table(index='cell_index', columns='step_col', values='volt', aggfunc='first')
        pivot = pivot.reset_index()

        try:
            if self.cell_index_col in df_cell.columns:
                target_dtype = df_cell[self.cell_index_col].dtype
                pivot['cell_index'] = pivot['cell_index'].astype(target_dtype)
        except Exception:
            self.logger.debug("Could not cast pivot.cell_index to df_cell cell_index dtype; merging with best effort")

        if self.cell_index_col not in df_cell.columns:
            raise KeyError(f"cell-level DataFrame must contain column '{self.cell_index_col}'")

        df_merged = df_cell.merge(pivot, left_on=self.cell_index_col, right_on='cell_index', how='left', validate='1:1')
        if 'cell_index' in df_merged.columns and self.cell_index_col != 'cell_index':
            df_merged.drop(columns=['cell_index'], inplace=True)

        return df_merged

def build_pack_features(
        df: pd.DataFrame,
        group_col: str = "pack_code",
        numeric_cols: Optional[List[str]] = None,
        step_prefix: str = "step_",
        step_range_for_inputs: range = range(1, 10),  # 1..9 inclusive
        stats: Optional[List[str]] = None,
        include_counts: bool = True
) -> pd.DataFrame:
    """
    Build pack-level aggregated features for tree models and compute separate targets:
      - target_step{t}_diff = max(step_t across cells in pack) - min(step_t across cells in pack)
    for each t in target_steps.

    Parameters
    ----------
    df : pd.DataFrame
        Cell-level dataframe containing pack_code and numeric columns.
    group_col : str
        Column to group by (default "pack_code").
    numeric_cols : Optional[List[str]]
        Columns to aggregate. If None, defaults to ['capacity','ocv3','ocv4','acr3','acr4','k_value','cell_thickness','weight']
        plus step_1_volt..step_9_volt.
    step_prefix : str
        Prefix used for step voltage columns.
    step_range_for_inputs : range
        Range of step numbers to include as input features.
    target_steps : List[int]
        Steps for which separate targets will be computed (default [14,15]).
    stats : Optional[List[str]]
        Stats to compute for each numeric column. Allowed: 'mean','std','min','max','median','q25','q75','range'.
        Default: all of them.
    include_counts : bool
        Whether to include n_cells per pack.
    save_path : Optional[str]
        If provided, save resulting dataframe to CSV at this path.

    Returns
    -------
    pack_df : pd.DataFrame
        One row per pack with aggregated features and separate target columns:
          - step{t}_max, step{t}_min, target_step{t}_diff for each t in target_steps.
    """
    default_base_cols = ['capacity', 'ocv3', 'ocv4', 'acr3', 'acr4', 'k_value', 'cell_thickness', 'weight']
    step_input_cols = [f"{step_prefix}{i}_volt" for i in step_range_for_inputs]
    if numeric_cols is None:
        numeric_cols = default_base_cols + step_input_cols
    else:
        for c in step_input_cols:
            if c not in numeric_cols:
                numeric_cols.append(c)
    if stats is None:
        stats = ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range']
    allowed_stats = {'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range'}
    for s in stats:
        if s not in allowed_stats:
            raise ValueError(f"Unsupported stat '{s}'. Allowed: {allowed_stats}")
    dfc = df.copy()
    for c in numeric_cols:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors='coerce')
        else:
            dfc[c] = np.nan

    def q25(x):
        return np.nanpercentile(x, 25) if len(x) > 0 else np.nan

    def q75(x):
        return np.nanpercentile(x, 75) if len(x) > 0 else np.nan

    rows = []
    grouped = dfc.groupby(group_col)
    for pack, g in grouped:
        out = {group_col: pack}
        if 'vehicle_code' in g.columns:
            out['vehicle_code'] = g['vehicle_code'].iloc[0]
        if include_counts:
            out['n_cells'] = len(g)
        for col in numeric_cols:
            vals = g[col].dropna().values
            if len(vals) == 0:
                for s in stats:
                    out[f"{col}_{s}"] = np.nan
                continue
            if 'mean' in stats: out[f"{col}_mean"] = float(np.mean(vals))
            if 'std' in stats: out[f"{col}_std"] = float(np.std(vals, ddof=0))
            if 'min' in stats: out[f"{col}_min"] = float(np.min(vals))
            if 'max' in stats: out[f"{col}_max"] = float(np.max(vals))
            if 'median' in stats: out[f"{col}_median"] = float(np.median(vals))
            if 'q25' in stats: out[f"{col}_q25"] = float(q25(vals))
            if 'q75' in stats: out[f"{col}_q75"] = float(q75(vals))
            if 'range' in stats: out[f"{col}_range"] = float(np.max(vals) - np.min(vals))
        rows.append(out)
    pack_df = pd.DataFrame(rows)
    return pack_df


