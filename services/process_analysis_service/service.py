from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from typing import Optional
from .util import dtw_distance, dtw_similarity

class ProcessService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.table = 'jz2_pack_process_data'

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "ProcessService", "ready": self._ready}

    def _build_in_params(self, prefix: str, values: List[str]):
        params = {}
        placeholders = []
        for i, v in enumerate(values):
            key = f"{prefix}{i}"
            placeholders.append(f":{key}")
            params[key] = v
        return ",".join(placeholders), params

    def fetch_minute_downsampled_voltage_series(
            self,
            table_name: str,
            current_pack_code: str,
            acquire_time_col: str = "acquire_time",
            total_time_col: str = "total_time",
            value_col: str = "pack_voltage_range",
            start_marker: str = "00:00:01.000",
            lookback: int = 100,
            minute_agg: str = "mean",  # supported: "mean","min","max"
            pct_pairs: Optional[List[Tuple[float, float]]] = None,
            sigma_multiples: Optional[List[float]] = None,
            min_packs_per_min: int = 3
    ) -> pd.DataFrame:
        """
        SQL-layer per-minute downsample and multiple bounds.

        - pct_pairs: list of (low_pct, high_pct) pairs, e.g. [(1,99),(5,95)]
        - sigma_multiples: list of sigma multipliers, e.g. [2.0, 3.0] -> ±2σ, ±3σ

        Returns DataFrame indexed by minute_bin (int) with columns:
          - 'elapsed_td','current'
          - for each pct pair: 'lower_pct_{low}_{high}', 'upper_pct_{low}_{high}'
          - for each sigma: 'lower_sigma_{k}','upper_sigma_{k}'
        """
        pct_pairs = pct_pairs or []
        sigma_multiples = sigma_multiples or []

        sql_current_start = text(f"""
            SELECT {acquire_time_col} AS acquire_time
            FROM `{table_name}`
            WHERE pack_code = :current_pack_code
            AND {total_time_col} = :start_marker
            ORDER BY {acquire_time_col} DESC
            LIMIT 1
        """)

        try:
            cur_row = self.db_client.read_sql(sql_current_start, params={
                "current_pack_code": current_pack_code,
                "start_marker": start_marker
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        if cur_row.empty:
            raise HTTPException(status_code=404, detail=f"未找到 pack {current_pack_code} 的起始行 (total_time={start_marker})")

        current_acquire_time = cur_row['acquire_time'].iloc[0]

        sql_start_rows = text(f"""
            SELECT pack_code, {acquire_time_col} AS acquire_time
            FROM `{table_name}`
            WHERE {total_time_col} = :start_marker
            AND {acquire_time_col} < :current_acquire_time
            ORDER BY {acquire_time_col} DESC
            LIMIT :limit
        """)

        try:
            start_rows = self.db_client.read_sql(sql_start_rows, params={
                "start_marker": start_marker,
                "current_acquire_time": current_acquire_time,
                "limit": int(lookback)
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"历史pack码数据：数据库查询失败: {e}")

        if start_rows.empty:
            raise HTTPException(status_code=404, detail=f"未找到 pack {current_pack_code} 对应历史pack数据")

        pack_codes = start_rows['pack_code'].astype(str).tolist()

        if current_pack_code not in pack_codes:
            pack_codes.insert(0, current_pack_code)
            pack_codes = pack_codes[: (lookback + 1)]

        agg_map = {"mean": "AVG", "min": "MIN", "max": "MAX"}
        if minute_agg not in agg_map:
            raise HTTPException(status_code=500, detail="minute_agg must be one of 'mean','min','max' for SQL-agg")

        agg_expr = agg_map[minute_agg]

        placeholders, in_params = self._build_in_params("pc", pack_codes)
        sql_agg = text(f"""
            SELECT
                pack_code,
                FLOOR( TIME_TO_SEC( LEFT(TRIM({total_time_col}), 8) ) / 60 ) AS minute_bin,
                {agg_expr}({value_col}) AS val,
                COUNT(1) AS cnt_rows_per_group
            FROM `{table_name}`
            WHERE pack_code IN ({placeholders})
              AND {total_time_col} IS NOT NULL
              AND {total_time_col} != ''
            GROUP BY pack_code, minute_bin
            ORDER BY pack_code ASC, minute_bin ASC
            """)

        try:
            df_agg = self.db_client.read_sql(sql_agg, params=in_params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"历史过程数据：数据库查询失败: {e}")

        if df_agg.empty:
            raise HTTPException(status_code=404, detail="SQL 分钟聚合未返回任何行，检查表/列/pack_code 是否正确")

        pivot = df_agg.pivot(index='minute_bin', columns='pack_code', values='val').sort_index()
        counts = df_agg.pivot(index='minute_bin', columns='pack_code', values='cnt_rows_per_group').fillna(0).astype(
            int)

        pivot = pivot.dropna(axis=1, how='all')
        if pivot.shape[1] == 0:
            raise HTTPException(status_code=404, detail="按分钟聚合后所有 pack 在 val 上均为空")

        common_index = pivot.index.unique().sort_values()
        elapsed_idx = pd.to_timedelta(common_index.astype(int) * 60, unit='s')

        result_df = pd.DataFrame(index=common_index)
        result_df['elapsed_td'] = elapsed_idx
        result_df['current'] = pivot[current_pack_code].reindex(common_index).values

        for low_pct, high_pct in pct_pairs:
            q_low = float(low_pct) / 100.0
            q_high = float(high_pct) / 100.0
            lower_name = f"lower_pct_{int(low_pct)}_{int(high_pct)}"
            upper_name = f"upper_pct_{int(low_pct)}_{int(high_pct)}"

            lower = pivot.apply(lambda row: np.nan if (row.notna().sum() < min_packs_per_min)
            else float(np.nanquantile(row.dropna().values, q_low)), axis=1)
            upper = pivot.apply(lambda row: np.nan if (row.notna().sum() < min_packs_per_min)
            else float(np.nanquantile(row.dropna().values, q_high)), axis=1)

            result_df[lower_name] = lower.reindex(common_index).values
            result_df[upper_name] = upper.reindex(common_index).values

        if sigma_multiples:
            mean_per_min = pivot.apply(lambda row: np.nan if (row.notna().sum() < min_packs_per_min)
            else float(np.nanmean(row.dropna().values)), axis=1)
            std_per_min = pivot.apply(lambda row: np.nan if (row.notna().sum() < min_packs_per_min)
            else float(np.nanstd(row.dropna().values, ddof=1) if row.dropna().size > 1 else 0.0),
                                      axis=1)
            for k in sigma_multiples:
                lower_name = f"lower_sigma_{int(k)}"
                upper_name = f"upper_sigma_{int(k)}"
                lower = (mean_per_min - k * std_per_min).reindex(common_index).values
                upper = (mean_per_min + k * std_per_min).reindex(common_index).values
                result_df[lower_name] = lower
                result_df[upper_name] = upper

        result_df.attrs['pivot_df'] = pivot
        result_df.attrs['counts_df'] = counts
        result_df.attrs['pack_codes_used'] = list(pivot.columns)
        result_df.attrs['current_acquire_time'] = current_acquire_time
        result_df.attrs['minute_agg'] = minute_agg
        result_df.attrs['pct_pairs'] = pct_pairs
        result_df.attrs['sigma_multiples'] = sigma_multiples
        result_df.attrs['min_packs_per_min'] = min_packs_per_min

        return result_df

    def analyze_pack_timeseries(self,
            result_df: pd.DataFrame,
            current_col: str = "current",
            lb_col: str = "lower_pct_1_99",
            ub_col: str = "upper_pct_1_99",
            td_col: str = 'elapsed_td',
            over_ratio_thresh: float = 0.3,
            max_continuous_over_s: float = 300.0,
            dtw_sim_thresh: float = 0.3):

        df = result_df.copy()
        # 1) 基础字段准备
        df['t_sec'] = df[td_col].dt.total_seconds()

        cur = df[current_col].astype(float)
        lb = df[lb_col].astype(float)
        ub = df[ub_col].astype(float)

        center_line = (lb + ub) / 2

        dev = cur - center_line

        # 2) DTW指标计算
        dist, path = dtw_distance(center_line.fillna(method='ffill').values,
                                  cur.fillna(method='ffill').values)
        sim = dtw_similarity(dist, center_line, cur)

        # 3) 越界相关指标
        within = (cur >= lb) & (cur <= ub)
        df['over'] = ~within

        over_ratio = df['over'].mean()

        df['over_shift'] = df['over'].shift(1, fill_value=False)
        df['start_of_over'] = (df['over'] == True) & (df['over_shift'] == False)

        durations = []
        start_time = None

        for over, t in zip(df['over'], df['t_sec']):
            if over and start_time is None:
                start_time = t
            if not over and start_time is not None:
                durations.append(t - start_time)
                start_time = None

        if start_time is not None:
            durations.append(df['t_sec'].iloc[-1] - start_time)

        max_continuous_over = max(durations) if durations else 0.0

        area_above = 0.0
        area_below = 0.0
        if len(df) >= 2:
            t = df['t_sec'].values
            # 使用分段梯形积分（按区间中点高度）
            dt = np.diff(t)
            mid_dev = 0.5 * (dev[:-1] + dev[1:])
            for di, mdi in zip(dt, mid_dev):
                if mdi > 0:
                    area_above += mdi * di
                elif mdi < 0:
                    area_below += abs(mdi) * di

        # 7) 综合异常判定
        abnormal = (
                (over_ratio > over_ratio_thresh) and
                (max_continuous_over > max_continuous_over_s) and
                (sim < dtw_sim_thresh)
        )

        result = {
            "over_ratio": over_ratio,
            "max_continuous_over_s": max_continuous_over,
            "dtw_distance": dist,
            "dtw_similarity": sim,
            "area_above": area_above,
            "area_below": area_below,
            "is_abnormal_pack": abnormal,
        }
        return result

    def prepare_heatmap_json(
            self,
            result_df: pd.DataFrame,
            current_col: str = "current",
            lb_col: str = "lower_pct_1_99",
            ub_col: str = "upper_pct_1_99",
            center_zero: bool = True,
            round_ndigits: Optional[int] = 3
    ) -> Dict[str, Any]:
        """
          {
            "columns": [<x labels (str)>],
            "matrix": [[v0, v1, ...]],   # 1 x T
            "vmin": <float or null>,
            "vmax": <float or null>
          }
        """
        df = result_df.copy()
        if current_col not in df.columns or lb_col not in df.columns or ub_col not in df.columns:
            raise KeyError("需要列 current/lower_bound/upper_bound 在 result_df 中")

        # 计算偏差
        center_line = (df[lb_col].astype(float) + df[ub_col].astype(float)) / 2.0
        dev = df[current_col].astype(float) - center_line

        cols = [int(x) for x in df.index]

        # 1 x T 矩阵
        arr = dev.values.reshape(1, -1)
        if round_ndigits is not None:
            arr = np.round(arr.astype(float), round_ndigits)

        matrix: List[List[Optional[float]]] = []
        row: List[Optional[float]] = []
        for v in arr.ravel():
            if np.isnan(v) or not np.isfinite(v):
                row.append(None)
            else:
                row.append(float(v))
        matrix.append(row)

        finite_vals = [x for x in row if x is not None]
        if finite_vals:
            minv = float(min(finite_vals))
            maxv = float(max(finite_vals))
            if center_zero:
                absmax = max(abs(minv), abs(maxv))
                vmin, vmax = -absmax, absmax
            else:
                vmin, vmax = minv, maxv
        else:
            vmin, vmax = None, None

        payload = {
            "columns": cols,
            "matrix": matrix,
            "vmin": vmin,
            "vmax": vmax,
        }
        return payload

    def prepare_pack_timeseries_json(
            self,
            result_df: pd.DataFrame,
            current_col: str = "current",
            round_ndigits: Optional[int] = 3,
    ) -> Dict[str, Any]:
        """
        返回结构：
        {
          "columns": [<x label strings>],
          "current": [float|null,...],
          "bands": [
             {"label": "1-99%", "lower":[...], "upper":[...]},
             ...
          ],
          "meta": { "n_points": T }
        }
        """
        df = result_df.copy()
        if current_col not in df.columns:
            raise KeyError("result_df 必须包含 current 列")

        idx = df.index
        columns = [int(x) for x in idx]

        # --- bound pairs ---
        bound_pairs: List[Tuple[str, str, str]] = []
        for col in df.columns:
            if col.startswith("lower_"):
                suffix = col[len("lower_"):]
                up_col = f"upper_{suffix}"
                if up_col in df.columns:
                    bound_pairs.append((suffix, f"lower_{suffix}", up_col))

        # --- current 数组 ---
        cur_arr = df[current_col].astype(float).replace([np.inf, -np.inf], np.nan).values
        if round_ndigits is not None:
            cur_arr = np.round(cur_arr, round_ndigits)
        current = [None if (not np.isfinite(x)) else float(x) for x in cur_arr]

        # --- bands 数组 ---
        bands = []
        for i, (suffix, low_col, up_col) in enumerate(bound_pairs):
            lower = df[low_col].astype(float).replace([np.inf, -np.inf], np.nan).values
            upper = df[up_col].astype(float).replace([np.inf, -np.inf], np.nan).values
            if round_ndigits is not None:
                lower = np.round(lower, round_ndigits)
                upper = np.round(upper, round_ndigits)
            lower_list = [None if (not np.isfinite(x)) else float(x) for x in lower]
            upper_list = [None if (not np.isfinite(x)) else float(x) for x in upper]

            if suffix.startswith("pct_"):
                parts = suffix.split("_")
                label = f"{parts[1]}-{parts[2]}%" if len(parts) >= 3 else suffix
            elif suffix.startswith("sigma_"):
                parts = suffix.split("_")
                label = f"±{parts[1]}σ" if len(parts) >= 2 else suffix
            else:
                label = suffix

            bands.append({
                "label": label,
                "lower": lower_list,
                "upper": upper_list,
            })

        payload = {
            "columns": columns,
            "current": current,
            "bands": bands,
            "meta": {
                "n_points": int(len(columns)),
                "n_bands": len(bands)
            }
        }
        return payload

    def pack_process_analysis(self, pack_code: str) -> Dict[str, Any]:
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        result_df = self.fetch_minute_downsampled_voltage_series(
            table_name=self.table,
            current_pack_code=pack_code,
            pct_pairs=[(5, 95), (1, 99)],
            sigma_multiples=[],
            lookback=100,
            minute_agg="mean",
            min_packs_per_min=3
        )

        result = {}

        analysis_res = self.analyze_pack_timeseries(
            result_df,
            current_col='current',
            lb_col='lower_pct_1_99',
            ub_col='upper_pct_1_99',
            td_col='elapsed_td',
            over_ratio_thresh=0.3,
            max_continuous_over_s=300.0,
            dtw_sim_thresh=0.4
        )
        result.update({"analysis": analysis_res})

        heatmap_json = self.prepare_heatmap_json(
            result_df,
            current_col="current",
            lb_col="lower_pct_1_99",
            ub_col="upper_pct_1_99",
            center_zero=False
        )

        result.update({"deviation_heatmap": heatmap_json})

        timeseries_json = self.prepare_pack_timeseries_json(
            result_df,
            current_col="current"
        )

        result.update({"timeseries": timeseries_json})

        return result




