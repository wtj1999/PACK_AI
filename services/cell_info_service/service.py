import pandas as pd
import numpy as np
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from typing import Dict, List, Any, Optional, Tuple
from .util import extract_series_count


class CellinfoService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.table = 'jz2_pack_result_data'
        self.table1 = 'jz2_pack_cell_data'

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "CellinfoService", "ready": self._ready}

    def cell_info_return(self, pack_codes: List[str]):
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        sql = text(f"""
                        SELECT *
                        FROM `{self.table}`
                        WHERE pack_code IN :pack_codes
                    """)

        try:
            df = self.db_client.read_sql(sql, params={"pack_codes": pack_codes})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        n_cells = extract_series_count(df.iloc[0]['vehicle_to_pack_num'])
        if n_cells != 102 and n_cells != 96:
            raise HTTPException(status_code=404, detail="电芯数目配置数据读取有误")

        cell_sql = text(f"""
                                SELECT pack_code, cell_code, ocv4_time, module_in_pack, cell_in_module, capacity, ocv3, ocv4, acr3, acr4, k_value, cell_thickness, weight
                                FROM `{self.table1}`
                                WHERE pack_code IN :pack_codes
                            """)

        try:
            cell_df = self.db_client.read_sql(cell_sql, params={"pack_codes": pack_codes})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"电芯位置数据库查询失败: {e}")

        cell_df = cell_df.query("cell_code.notna() and cell_code != ''")
        cell_df = cell_df.sort_values('ocv4_time').drop_duplicates(
            subset=['pack_code', 'cell_code'], keep='last')

        if n_cells == 102:
            cell_map_df = pd.read_csv('services/dcr_analysis_service/data/cell_position_map.csv')
        else:
            cell_map_df = pd.read_csv('services/dcr_analysis_service/data/cell_position_map_96.csv')

        cell_df = pd.merge(cell_df, cell_map_df, how='left', on=['module_in_pack', 'cell_in_module'])
        cell_df = cell_df.sort_values(by=['pack_code', 'cell_index'])

        out_cols = [
            'pack_code', 'cell_code', 'module_in_pack', 'cell_in_module',
            'capacity', 'ocv3', 'ocv4', 'acr3', 'acr4', 'k_value', 'cell_thickness', 'weight',
            'cell_index'
        ]

        records = cell_df[out_cols].to_dict(orient='records')
        return {'results': records}