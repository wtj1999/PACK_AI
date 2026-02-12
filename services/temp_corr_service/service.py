from typing import Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import text
from .utils import _align_dicts_to_arrays, safe_corr_vec
from services.base import BaseService
from fastapi import HTTPException
from typing import List, Optional, Any
from datetime import datetime, timedelta


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
        self.result_table = "jz2_pack_result_data"
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

    def seek_pack_time(self, vehicle_code: str):
        sql = text(f"""
                            SELECT *
                            FROM `{self.result_table}`
                            WHERE vehicle_code = :vehicle_code
                            ORDER BY {self.pack_code_col}, {self.time_col}
                        """)
        params = {"vehicle_code": vehicle_code}
        df = self.db_client.read_sql(sql, params=params)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail="未查询到任何数据(vehicle_code)")

        return  str(pd.Timestamp(df["acquire_time"].iloc[0]) - timedelta(hours=1)), str(pd.Timestamp(df["acquire_time"].iloc[-1]) + timedelta(hours=1))


    def pack_temp_corr(self, vehicle_code: str, step_id: str) -> Dict[str, Any]:
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        strat_time, end_time = self.seek_pack_time(vehicle_code)

        if step_id == "all":
            sql = text(f"""
                    SELECT pack_code, step_id, step_name, acquire_time,
                    charge_energy, discharge_energy, charge_capacity, discharge_capacity,
                    bms_batttemp1, bms_batttemp2, bms_batttemp3, bms_batttemp4, bms_batttemp5,
                    bms_batttemp6, bms_batttemp7, bms_batttemp8
                    FROM `{self.table}`
                    WHERE vehicle_code = :vehicle_code
                    AND acquire_time BETWEEN :start_time AND :end_time
                    ORDER BY {self.pack_code_col}, {self.time_col}
                """)
            params = {"vehicle_code": vehicle_code, "start_time": strat_time, "end_time": end_time}
        else:
            sql = text(f"""
                    SELECT pack_code, step_id, step_name, acquire_time,
                    charge_energy, discharge_energy, charge_capacity, discharge_capacity,
                    bms_batttemp1, bms_batttemp2, bms_batttemp3, bms_batttemp4, bms_batttemp5,
                    bms_batttemp6, bms_batttemp7, bms_batttemp8
                    FROM `{self.table}`
                    WHERE vehicle_code = :vehicle_code
                      AND acquire_time BETWEEN :start_time AND :end_time
                      AND step_id = :step_id
                    ORDER BY {self.pack_code_col}, {self.time_col}
                """)
            params = {"vehicle_code": vehicle_code, "start_time": strat_time, "end_time": end_time, "step_id": step_id}

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
                t_key = str(pd.Timestamp(row["acquire_time"]))
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

        all_times_keys = sorted(
            set(list(temp_min_by_time.keys()) +
                list(charge_energy_by_time.keys()) +
                list(discharge_energy_by_time.keys()) +
                list(charge_capacity_by_time.keys()) +
                list(discharge_capacity_by_time.keys())
                )
        )

        temp_min_by_time = dict(sorted(temp_min_by_time.items(), key=lambda x: x[0]))

        charge_energy_by_time = dict(sorted(charge_energy_by_time.items(), key=lambda x: x[0]))
        discharge_energy_by_time = dict(sorted(discharge_energy_by_time.items(), key=lambda x: x[0]))
        charge_capacity_by_time = dict(sorted(charge_capacity_by_time.items(), key=lambda x: x[0]))
        discharge_capacity_by_time = dict(sorted(discharge_capacity_by_time.items(), key=lambda x: x[0]))

        temp_min_list = [{"time": t, "temp_min": temp_min_by_time.get(t, None)} for t in all_times_keys]
        charge_energy_list = [{"time": t, "charge_energy": charge_energy_by_time.get(t, None)} for t in all_times_keys]
        discharge_energy_list = [{"time": t, "discharge_energy": discharge_energy_by_time.get(t, None)} for t in
                                 all_times_keys]
        charge_capacity_list = [{"time": t, "charge_capacity": charge_capacity_by_time.get(t, None)} for t in
                                all_times_keys]
        discharge_capacity_list = [{"time": t, "discharge_capacity": discharge_capacity_by_time.get(t, None)} for t in
                                   all_times_keys]

        a_temp, a_charge_energy, _ = _align_dicts_to_arrays(temp_min_by_time, charge_energy_by_time)
        _, a_discharge_energy, _ = _align_dicts_to_arrays(temp_min_by_time, discharge_energy_by_time)
        _, a_charge_capacity, _ = _align_dicts_to_arrays(temp_min_by_time, charge_capacity_by_time)
        _, a_discharge_capacity, _ = _align_dicts_to_arrays(temp_min_by_time, discharge_capacity_by_time)

        corr_charge_energy = safe_corr_vec(a_temp, a_charge_energy)
        corr_discharge_energy = safe_corr_vec(a_temp, a_discharge_energy)
        corr_charge_capacity = safe_corr_vec(a_temp, a_charge_capacity)
        corr_discharge_capacity = safe_corr_vec(a_temp, a_discharge_capacity)

        corr_energy = safe_corr_vec(a_temp, a_charge_energy + a_discharge_energy)
        corr_capacity = safe_corr_vec(a_temp, a_charge_capacity + a_discharge_capacity)

        if step_name is not None:
            if '充电' in step_name:
                corr_energy = corr_charge_energy
                corr_capacity = corr_charge_capacity
            elif '放电' in step_name:
                corr_energy = corr_discharge_energy
                corr_capacity = corr_discharge_capacity
            else:
                corr_energy = None
                corr_capacity = None
        else:
            corr_energy = corr_energy
            corr_capacity = corr_capacity


        result = {
            "vehicle_code": vehicle_code,
            "step_id": step_id,
            "corr_minTemp_energy": corr_energy,
            "corr_minTemp_capacity": corr_capacity,
            "temp_min_list": temp_min_list,
            "charge_energy_list": charge_energy_list,
            "discharge_energy_list": discharge_energy_list,
            "charge_capacity_list": charge_capacity_list,
            "discharge_capacity_list": discharge_capacity_list
        }

        return result




