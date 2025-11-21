from typing import Dict, Any
import pandas as pd
from sqlalchemy import text
from .utils import safe_corr, series_to_pylist
from services.base import BaseService
from fastapi import HTTPException


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
        self.stage_time_col = "stage_time"
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

    def pack_temp_corr(self, vehicle_code: str, step_id: str) -> Dict[str, Any]:
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        sql = text(f"""
            SELECT *
            FROM `{self.table}`
            WHERE vehicle_code = :vehicle_code
              AND step_id = :step_id
            ORDER BY {self.pack_code_col}, {self.stage_time_col}
        """)

        try:
            df = self.db_client.read_sql(sql, params={"vehicle_code": vehicle_code, "step_id": step_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        if df.empty:
            raise HTTPException(status_code=404, detail="未查询到任何数据（vehicle_code/step_id 可能不匹配）")

        required_cols = {self.pack_code_col, self.stage_time_col, self.charge_energy_col, self.charge_capacity_col,
                         self.discharge_energy_col, self.discharge_capacity_col, 'step_name'}
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=500, detail=f"缺少期望的列：{missing_cols}")

        valid_step_names = ["静置", "恒流充电", "恒流放电"]
        step_name_vals = df['step_name'].dropna().unique().tolist()
        step_name = None
        for name in step_name_vals:
            if name in valid_step_names:
                step_name = name
                break

        df = df[df['step_name'] == step_name]

        pack_codes = df[self.pack_code_col].dropna().unique().tolist()
        if not pack_codes:
            raise HTTPException(status_code=500, detail="没有找到任何 pack_code")
        pack_codes.sort()

        pack_dfs: Dict[str, pd.DataFrame] = {}
        for p in pack_codes:
            sub = df[df[self.pack_code_col] == p].copy()
            sub = sub.sort_values(self.stage_time_col).reset_index(drop=True)
            pack_dfs[p] = sub

        lengths = [len(g) for g in pack_dfs.values()]
        min_len = min(lengths) if lengths else 0
        if min_len == 0:
            raise HTTPException(status_code=500, detail="至少有一个 pack 没有时间序列数据")

        temp_concat_list = []
        meta_frames = []
        for idx, p in enumerate(pack_codes):
            g = pack_dfs[p].iloc[:min_len].reset_index(drop=True)
            missing_temp_cols = [c for c in self.temp_cols_per_pack if c not in g.columns]
            if missing_temp_cols:
                raise HTTPException(status_code=500, detail=f"pack {p} 缺少温度列：{missing_temp_cols}")

            offset = idx * len(self.temp_cols_per_pack)
            rename_map = {old: f'BMS_BattTemp{offset + i + 1}' for i, old in enumerate(self.temp_cols_per_pack)}
            tdf_renamed = g[self.temp_cols_per_pack].rename(columns=rename_map).reset_index(drop=True)
            temp_concat_list.append(tdf_renamed)

            meta_frames.append(g[[self.stage_time_col, self.charge_energy_col, self.charge_capacity_col,
                                  self.discharge_energy_col, self.discharge_capacity_col]].reset_index(drop=True))

        temps_df = pd.concat(temp_concat_list, axis=1)
        if temps_df.empty:
            raise HTTPException(status_code=500, detail="合并后的温度表为空")

        temps_df = temps_df.apply(pd.to_numeric, errors='coerce')
        min_temps = temps_df.min(axis=1, skipna=True)

        base_meta = meta_frames[0]
        charge_energy = pd.to_numeric(base_meta[self.charge_energy_col], errors='coerce')
        charge_capacity = pd.to_numeric(base_meta[self.charge_capacity_col], errors='coerce')
        discharge_energy = pd.to_numeric(base_meta[self.discharge_energy_col], errors='coerce')
        discharge_capacity = pd.to_numeric(base_meta[self.discharge_capacity_col], errors='coerce')

        # 根据 step_name 决定使用充电还是放电数据计算相关性
        if step_name and '充电' in str(step_name):
            corr_with_energy = safe_corr(min_temps, charge_energy)
            corr_with_capacity = safe_corr(min_temps, charge_capacity)
            energy_series = charge_energy
            capacity_series = charge_capacity
        else:
            corr_with_energy = safe_corr(min_temps, discharge_energy)
            corr_with_capacity = safe_corr(min_temps, discharge_capacity)
            energy_series = discharge_energy
            capacity_series = discharge_capacity

        result = {
            "vehicle_code": vehicle_code,
            "step_id": step_id,
            "step_name": step_name,
            "corr_minTemp_energy": corr_with_energy,
            "corr_minTemp_capacity": corr_with_capacity,
            "min_temp_list": series_to_pylist(min_temps),
            "energy_list": series_to_pylist(energy_series),
            "capacity_list": series_to_pylist(capacity_series),
        }

        return result
