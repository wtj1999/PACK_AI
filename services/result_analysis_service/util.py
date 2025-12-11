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
            return set(range(1, 16))
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

