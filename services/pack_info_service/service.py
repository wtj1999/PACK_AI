import logging
import torch
import re
import pandas as pd
import numpy as np
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from typing import Dict, List, Any, Optional, Tuple



class PackinfoService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.table = 'jz2_pack_result_data'
        self.step_id = '1'

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "PackinfoService", "ready": self._ready}

    def pack_info_return(self, payload):
        """
            Query pack info based on payload filters:
              - vehicleCode (exact match)
              - startTime / endTime (one or both supported)
              - elecProcessConfig (vehicle_to_pack_num equality if provided)
            Returns a pandas.DataFrame (or raises HTTPException on DB error).
            """
        vehicle_code = payload.get("vehicleCode")
        start_time = payload.get("startTime")
        end_time = payload.get("endTime")
        elec_config = payload.get("elecProcessConfig")

        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        conditions = ["step_id = :step_id"]
        params = {"step_id": str(self.step_id)}

        # vehicle_code filter (if provided)
        if vehicle_code:
            conditions.append("vehicle_code = :vehicle_code")
            params["vehicle_code"] = vehicle_code

        # time filters: support start only / end only / both
        if start_time and end_time:
            conditions.append("acquire_time BETWEEN :start_time AND :end_time")
            params["start_time"] = start_time
            params["end_time"] = end_time
        else:
            if start_time:
                conditions.append("acquire_time >= :start_time")
                params["start_time"] = start_time
            if end_time:
                conditions.append("acquire_time <= :end_time")
                params["end_time"] = end_time

        # elecProcessConfig filter (only when explicitly provided, allow 0/'' if needed)
        if elec_config:
            conditions.append("vehicle_to_pack_num = :elec_config")
            params["elec_config"] = elec_config

        where_clause = " AND ".join(conditions)

        sql = text(f"""
                SELECT pack_code, acquire_time, vehicle_code, vehicle_to_pack_num
                FROM `{self.table}`
                WHERE {where_clause}
                ORDER BY acquire_time ASC
            """)

        try:
            df = self.db_client.read_sql(sql, params=params)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        df_dedup = df.drop_duplicates(subset=['pack_code'])
        grouped = df_dedup.groupby('vehicle_code', sort=True)

        result: List[Dict] = []
        for vehicle_code, g in grouped:
            if pd.isna(vehicle_code):
                continue

            pack_list = sorted([str(x) for x in g['pack_code'].dropna().unique()])

            elec_str = [v for v in g['vehicle_to_pack_num'].dropna().unique()][0]
            date_str = str([v for v in g['acquire_time'].dropna().unique()][0])

            item = {
                "vehicleCode": str(vehicle_code),
                "elecProcessConfig": elec_str,
                "packCodeList": pack_list,
                "dateTime": date_str,
            }
            result.append(item)

        return {'results': result}

