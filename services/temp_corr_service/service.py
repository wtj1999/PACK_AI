from typing import Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import text
from .utils import _align_dicts_to_arrays, safe_corr_vec
from services.base import BaseService
from fastapi import HTTPException
from typing import List, Optional, Any


class TempService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.table = "jz2_pack_process_data"
        self.pack_code_col = "pack_code"
        self.time_col = "acquire_time"
        self.charge_energy_col = "charge_energy"
        self.charge_capacity_col = "charge_capacity"
        self.discharge_energy_col = "discharge_energy"
        self.discharge_capacity_col = "discharge_capacity"
        self.temp_cols_per_pack = [f"bms_batttemp{i}" for i in range(1, 9)]

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "TempService", "ready": self._ready}

    # def pack_temp_corr(self, vehicle_code: str, step_id: str) -> Dict[str, Any]:
    #     if self.db_client is None:
    #         raise HTTPException(status_code=500, detail="数据库引擎创建失败")
    #
    #     sql = text(f"""
    #         SELECT *
    #         FROM `{self.table}`
    #         WHERE vehicle_code = :vehicle_code
    #           AND step_id = :step_id
    #         ORDER BY {self.pack_code_col}, {self.stage_time_col}
    #     """)
    #
    #     try:
    #         df = self.db_client.read_sql(sql, params={"vehicle_code": vehicle_code, "step_id": step_id})
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")
    #
    #     if df.empty:
    #         raise HTTPException(status_code=404, detail="未查询到任何数据（vehicle_code/step_id 可能不匹配）")
    #
    #     required_cols = {self.pack_code_col, self.stage_time_col, self.charge_energy_col, self.charge_capacity_col,
    #                      self.discharge_energy_col, self.discharge_capacity_col, 'step_name'}
    #     missing_cols = [c for c in required_cols if c not in df.columns]
    #     if missing_cols:
    #         raise HTTPException(status_code=500, detail=f"缺少期望的列：{missing_cols}")
    #
    #     valid_step_names = ["静置", "恒流充电", "恒流放电"]
    #     step_name_vals = df['step_name'].dropna().unique().tolist()
    #     step_name = None
    #     for name in step_name_vals:
    #         if name in valid_step_names:
    #             step_name = name
    #             break
    #
    #     df = df[df['step_name'] == step_name]
    #
    #     pack_codes = df[self.pack_code_col].dropna().unique().tolist()
    #     if not pack_codes:
    #         raise HTTPException(status_code=500, detail="没有找到任何 pack_code")
    #     pack_codes.sort()
    #
    #     pack_dfs: Dict[str, pd.DataFrame] = {}
    #     for p in pack_codes:
    #         sub = df[df[self.pack_code_col] == p].copy()
    #         sub = sub.sort_values(self.stage_time_col).reset_index(drop=True)
    #         pack_dfs[p] = sub
    #
    #     lengths = [len(g) for g in pack_dfs.values()]
    #     min_len = min(lengths) if lengths else 0
    #     if min_len == 0:
    #         raise HTTPException(status_code=500, detail="至少有一个 pack 没有时间序列数据")
    #
    #     temp_concat_list = []
    #     meta_frames = []
    #     for idx, p in enumerate(pack_codes):
    #         g = pack_dfs[p].iloc[:min_len].reset_index(drop=True)
    #         missing_temp_cols = [c for c in self.temp_cols_per_pack if c not in g.columns]
    #         if missing_temp_cols:
    #             raise HTTPException(status_code=500, detail=f"pack {p} 缺少温度列：{missing_temp_cols}")
    #
    #         offset = idx * len(self.temp_cols_per_pack)
    #         rename_map = {old: f'BMS_BattTemp{offset + i + 1}' for i, old in enumerate(self.temp_cols_per_pack)}
    #         tdf_renamed = g[self.temp_cols_per_pack].rename(columns=rename_map).reset_index(drop=True)
    #         temp_concat_list.append(tdf_renamed)
    #
    #         meta_frames.append(g[[self.stage_time_col, self.charge_energy_col, self.charge_capacity_col,
    #                               self.discharge_energy_col, self.discharge_capacity_col]].reset_index(drop=True))
    #
    #     temps_df = pd.concat(temp_concat_list, axis=1)
    #     if temps_df.empty:
    #         raise HTTPException(status_code=500, detail="合并后的温度表为空")
    #
    #     temps_df = temps_df.apply(pd.to_numeric, errors='coerce')
    #     min_temps = temps_df.min(axis=1, skipna=True)
    #
    #     base_meta = meta_frames[0]
    #     charge_energy = pd.to_numeric(base_meta[self.charge_energy_col], errors='coerce')
    #     charge_capacity = pd.to_numeric(base_meta[self.charge_capacity_col], errors='coerce')
    #     discharge_energy = pd.to_numeric(base_meta[self.discharge_energy_col], errors='coerce')
    #     discharge_capacity = pd.to_numeric(base_meta[self.discharge_capacity_col], errors='coerce')
    #
    #     # 根据 step_name 决定使用充电还是放电数据计算相关性
    #     if step_name and '充电' in str(step_name):
    #         corr_with_energy = safe_corr(min_temps, charge_energy)
    #         corr_with_capacity = safe_corr(min_temps, charge_capacity)
    #         energy_series = charge_energy
    #         capacity_series = charge_capacity
    #     else:
    #         corr_with_energy = safe_corr(min_temps, discharge_energy)
    #         corr_with_capacity = safe_corr(min_temps, discharge_capacity)
    #         energy_series = discharge_energy
    #         capacity_series = discharge_capacity
    #
    #     result = {
    #         "vehicle_code": vehicle_code,
    #         "step_id": step_id,
    #         "step_name": step_name,
    #         "corr_minTemp_energy": corr_with_energy,
    #         "corr_minTemp_capacity": corr_with_capacity,
    #         "min_temp_list": series_to_pylist(min_temps),
    #         "energy_list": series_to_pylist(energy_series),
    #         "capacity_list": series_to_pylist(capacity_series),
    #     }
    #
    #     return result

    def _latest_contiguous_segment(self, df_pack: pd.DataFrame, time_col: str, gap_seconds: int = 3600) -> pd.DataFrame:
        if df_pack is None or df_pack.empty:
            return df_pack
        times = pd.to_datetime(df_pack[time_col])
        diffs = times.diff().dt.total_seconds().fillna(0)
        split_idx = np.where(diffs > gap_seconds)[0].tolist()
        if not split_idx:
            return df_pack
        last_split = split_idx[-1]
        return df_pack.iloc[last_split:].reset_index(drop=True)

    # def pack_temp_corr(self, vehicle_code: str, step_id: str) -> Dict[str, Any]:
    #     if self.db_client is None:
    #         raise HTTPException(status_code=500, detail="数据库引擎创建失败")
    #
    #     if step_id == "all":
    #         sql = text(f"""
    #                 SELECT *
    #                 FROM `{self.table}`
    #                 WHERE vehicle_code = :vehicle_code
    #                 ORDER BY {self.pack_code_col}, {self.time_col}
    #             """)
    #         params = {"vehicle_code": vehicle_code}
    #     else:
    #         sql = text(f"""
    #                 SELECT *
    #                 FROM `{self.table}`
    #                 WHERE vehicle_code = :vehicle_code
    #                   AND step_id = :step_id
    #                 ORDER BY {self.pack_code_col}, {self.time_col}
    #             """)
    #         params = {"vehicle_code": vehicle_code, "step_id": step_id}
    #
    #     try:
    #         df = self.db_client.read_sql(sql, params=params)
    #     except Exception as e:
    #         raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")
    #
    #     if df is None or df.empty:
    #         raise HTTPException(status_code=404, detail="未查询到任何数据（vehicle_code/step_id 可能不匹配）")
    #
    #     required_cols = {self.pack_code_col, self.time_col, self.charge_energy_col, self.charge_capacity_col,
    #                      self.discharge_energy_col, self.discharge_capacity_col, 'step_name'}
    #     missing_cols = [c for c in required_cols if c not in df.columns]
    #     if missing_cols:
    #         raise HTTPException(status_code=500, detail=f"缺少期望的列：{missing_cols}")
    #
    #     valid_step_names = ["静置", "恒流充电", "恒流放电"]
    #     step_name = None
    #     if step_id != "all":
    #         step_name_vals = df['step_name'].dropna().unique().tolist()
    #         for name in step_name_vals:
    #             if name in valid_step_names:
    #                 step_name = name
    #                 break
    #         if step_name is None:
    #             if step_name_vals:
    #                 step_name = step_name_vals[0]
    #         df = df[df['step_name'] == step_name]
    #
    #     pack_codes = df[self.pack_code_col].dropna().unique().tolist()
    #     if not pack_codes:
    #         raise HTTPException(status_code=500, detail="没有找到任何 pack_code")
    #     pack_codes.sort()
    #
    #     pack_dfs: Dict[str, pd.DataFrame] = {}
    #     gap_seconds = 3600
    #     for p in pack_codes:
    #         sub = df[df[self.pack_code_col] == p].copy()
    #         sub = sub.sort_values(self.time_col).reset_index(drop=True)
    #         latest_seg = self._latest_contiguous_segment(sub, self.time_col, gap_seconds=gap_seconds)
    #         if latest_seg is None or latest_seg.empty:
    #             continue
    #         pack_dfs[p] = latest_seg
    #
    #     if not pack_dfs:
    #         raise HTTPException(status_code=500, detail="按最新批次筛选后未找到有效 pack 数据")
    #
    #     lengths = [len(g) for g in pack_dfs.values()]
    #     min_len = min(lengths) if lengths else 0
    #     if min_len == 0:
    #         raise HTTPException(status_code=500, detail="至少有一个 pack 没有时间序列数据（最新批次）")
    #
    #     temp_concat_list = []
    #     meta_frames = []
    #     for idx, p in enumerate(sorted(pack_dfs.keys())):
    #         g = pack_dfs[p].iloc[:min_len].reset_index(drop=True)
    #         missing_temp_cols = [c for c in self.temp_cols_per_pack if c not in g.columns]
    #         if missing_temp_cols:
    #             raise HTTPException(status_code=500, detail=f"pack {p} 缺少温度列：{missing_temp_cols}")
    #
    #         offset = idx * len(self.temp_cols_per_pack)
    #         rename_map = {old: f'BMS_BattTemp{offset + i + 1}' for i, old in enumerate(self.temp_cols_per_pack)}
    #         tdf_renamed = g[self.temp_cols_per_pack].rename(columns=rename_map).reset_index(drop=True)
    #         temp_concat_list.append(tdf_renamed)
    #
    #         meta_frames.append(g[[self.time_col, self.charge_energy_col, self.charge_capacity_col,
    #                               self.discharge_energy_col, self.discharge_capacity_col]].reset_index(drop=True))
    #
    #     temps_df = pd.concat(temp_concat_list, axis=1)
    #     if temps_df.empty:
    #         raise HTTPException(status_code=500, detail="合并后的温度表为空")
    #
    #     temps_df = temps_df.apply(pd.to_numeric, errors='coerce')
    #     min_temps = temps_df.min(axis=1, skipna=True)
    #
    #     base_meta = meta_frames[0]
    #     charge_energy = pd.to_numeric(base_meta[self.charge_energy_col], errors='coerce')
    #     charge_capacity = pd.to_numeric(base_meta[self.charge_capacity_col], errors='coerce')
    #     discharge_energy = pd.to_numeric(base_meta[self.discharge_energy_col], errors='coerce')
    #     discharge_capacity = pd.to_numeric(base_meta[self.discharge_capacity_col], errors='coerce')
    #
    #     result: Dict[str, Any] = {
    #         "vehicle_code": vehicle_code,
    #         "step_id": step_id,
    #         "step_name": step_name,
    #         "min_temp_list": series_to_pylist(min_temps),
    #         "time_list": pd.to_datetime(base_meta[self.time_col]).tolist(),
    #     }
    #
    #     # if step_id != "all":
    #     #     if step_name and '充电' in str(step_name):
    #     #         corr_with_energy = safe_corr(min_temps, charge_energy)
    #     #         corr_with_capacity = safe_corr(min_temps, charge_capacity)
    #     #         result.update({
    #     #             "corr_minTemp_energy": corr_with_energy,
    #     #             "corr_minTemp_capacity": corr_with_capacity,
    #     #             "energy_list": series_to_pylist(charge_energy),
    #     #             "capacity_list": series_to_pylist(charge_capacity),
    #     #         })
    #     #     else:
    #     #         corr_with_energy = safe_corr(min_temps, discharge_energy)
    #     #         corr_with_capacity = safe_corr(min_temps, discharge_capacity)
    #     #         result.update({
    #     #             "corr_minTemp_energy": corr_with_energy,
    #     #             "corr_minTemp_capacity": corr_with_capacity,
    #     #             "energy_list": series_to_pylist(discharge_energy),
    #     #             "capacity_list": series_to_pylist(discharge_capacity),
    #     #         })
    #     # else:
    #     corr_charge_energy = safe_corr(min_temps, charge_energy)
    #     corr_charge_capacity = safe_corr(min_temps, charge_capacity)
    #     corr_discharge_energy = safe_corr(min_temps, discharge_energy)
    #     corr_discharge_capacity = safe_corr(min_temps, discharge_capacity)
    #
    #     result.update({
    #         "corr_minTemp_charge_energy": corr_charge_energy,
    #         "corr_minTemp_charge_capacity": corr_charge_capacity,
    #         "corr_minTemp_discharge_energy": corr_discharge_energy,
    #         "corr_minTemp_discharge_capacity": corr_discharge_capacity,
    #         "charge_energy_list": series_to_pylist(charge_energy),
    #         "charge_capacity_list": series_to_pylist(charge_capacity),
    #         "discharge_energy_list": series_to_pylist(discharge_energy),
    #         "discharge_capacity_list": series_to_pylist(discharge_capacity),
    #     })
    #
    #     return result

    def pack_temp_corr(self, vehicle_code: str, step_id: str) -> Dict[str, Any]:
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        if step_id == "all":
            sql = text(f"""
                    SELECT *
                    FROM `{self.table}`
                    WHERE vehicle_code = :vehicle_code
                    ORDER BY {self.pack_code_col}, {self.time_col}
                """)
            params = {"vehicle_code": vehicle_code}
        else:
            sql = text(f"""
                    SELECT *
                    FROM `{self.table}`
                    WHERE vehicle_code = :vehicle_code
                      AND step_id = :step_id
                    ORDER BY {self.pack_code_col}, {self.time_col}
                """)
            params = {"vehicle_code": vehicle_code, "step_id": step_id}

        try:
            df = self.db_client.read_sql(sql, params=params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="未查询到任何数据（vehicle_code/step_id 可能不匹配）")

        valid_step_names = ["静置", "恒流充电", "恒流放电"]
        step_name = None
        if step_id != "all":
            step_name_vals = df['step_name'].dropna().unique().tolist()
            for name in step_name_vals:
                if name in valid_step_names:
                    step_name = name
                    break
            if step_name is None:
                if step_name_vals:
                    step_name = step_name_vals[0]
            df = df[df['step_name'] == step_name]

        temp_min_by_time = {}
        charge_energy_by_time = {}
        discharge_energy_by_time = {}
        charge_capacity_by_time = {}
        discharge_capacity_by_time = {}

        bat_temp_cols = [f"bms_batttemp{i}" for i in range(1, 9)]

        for _, row in df.iterrows():
            try:
                t_key = pd.Timestamp(row["acquire_time"]).isoformat()
            except Exception:
                t_key = str(row["acquire_time"])

            tvals = pd.to_numeric(row[bat_temp_cols], errors="coerce").astype(float)
            tvals_nonan = tvals[~np.isnan(tvals)]
            if not tvals_nonan.empty:
                cur_min = float(tvals_nonan.min())
                if t_key in temp_min_by_time:
                    temp_min_by_time[t_key] = min(temp_min_by_time[t_key], cur_min)
                else:
                    temp_min_by_time[t_key] = cur_min

            if "charge_energy" in row.index and (t_key not in charge_energy_by_time):
                v = row.get("charge_energy")
                if pd.notna(v):
                    charge_energy_by_time[t_key] = float(v)
            if "discharge_energy" in row.index and (t_key not in discharge_energy_by_time):
                v = row.get("discharge_energy")
                if pd.notna(v):
                    discharge_energy_by_time[t_key] = float(v)
            if "charge_capacity" in row.index and (t_key not in charge_capacity_by_time):
                v = row.get("charge_capacity")
                if pd.notna(v):
                    charge_capacity_by_time[t_key] = float(v)
            if "discharge_capacity" in row.index and (t_key not in discharge_capacity_by_time):
                v = row.get("discharge_capacity")
                if pd.notna(v):
                    discharge_capacity_by_time[t_key] = float(v)

        temp_min_by_time = dict(sorted(temp_min_by_time.items(), key=lambda x: x[0]))

        charge_energy_by_time = dict(sorted(charge_energy_by_time.items(), key=lambda x: x[0]))
        discharge_energy_by_time = dict(sorted(discharge_energy_by_time.items(), key=lambda x: x[0]))
        charge_capacity_by_time = dict(sorted(charge_capacity_by_time.items(), key=lambda x: x[0]))
        discharge_capacity_by_time = dict(sorted(discharge_capacity_by_time.items(), key=lambda x: x[0]))

        a_temp, a_charge_energy, _ = _align_dicts_to_arrays(temp_min_by_time, charge_energy_by_time)
        _, a_discharge_energy, _ = _align_dicts_to_arrays(temp_min_by_time, discharge_energy_by_time)
        _, a_charge_capacity, _ = _align_dicts_to_arrays(temp_min_by_time, charge_capacity_by_time)
        _, a_discharge_capacity, _ = _align_dicts_to_arrays(temp_min_by_time, discharge_capacity_by_time)

        corr_charge_energy = safe_corr_vec(a_temp, a_charge_energy)
        corr_discharge_energy = safe_corr_vec(a_temp, a_discharge_energy)
        corr_charge_capacity = safe_corr_vec(a_temp, a_charge_capacity)
        corr_discharge_capacity = safe_corr_vec(a_temp, a_discharge_capacity)

        result = {
            "vehicle_code": vehicle_code,
            "step_id": step_id,
            "corr_minTemp_charge_energy": corr_charge_energy,
            "corr_minTemp_charge_capacity": corr_charge_capacity,
            "corr_minTemp_discharge_energy": corr_discharge_energy,
            "corr_minTemp_discharge_capacity": corr_discharge_capacity,
            "charge_energy_by_time": charge_energy_by_time,
            "discharge_energy_by_time": discharge_energy_by_time,
            "charge_capacity_by_time": charge_capacity_by_time,
            "discharge_capacity_by_time": discharge_capacity_by_time
        }

        return result




