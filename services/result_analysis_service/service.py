from typing import Dict, List, Any
import pandas as pd
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException


class ResultService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.table = 'jz2_pack_result_data'

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "ResultService", "ready": self._ready}

    def pack_result_analysis(self, pack_code: str) -> List:
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        sql = text(f"""
                SELECT *
                FROM `{self.table}`
                WHERE pack_code = :pack_code
                AND step_id IN :step_id
            """)

        try:
            df = self.db_client.read_sql(sql, params={"pack_code": pack_code, "step_id": ['1', '8', '9', '14', '15']})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        if len(df) < 5:
            raise HTTPException(status_code=404, detail="数据库中结果数据不完整")

        volt_cols = [c for c in df.columns if c and c.startswith("bms_cellvolt")]
        stepName_list = ['测前电压', '充电末端动态电压', '充电后静态电压', '放电末端动态电压', '放电后静态电压']
        cell_map_df = pd.read_csv('services/result_analysis_service/data/cell_position_map.csv')
        result_list = []

        for i, id in enumerate(['1', '8', '9', '14', '15']):
            volt_data = df[df['step_id'] == id][volt_cols].iloc[0].values
            volt_df = pd.DataFrame({
                'cell_index': range(1, len(volt_cols) + 1),
                'cell_volt': volt_data
            })
            volt_df = pd.merge(volt_df, cell_map_df, how='left', left_on='cell_index', right_on='cell_index')
            vals = volt_df.sort_values(['module_in_pack', 'cell_in_module'])['cell_volt'].values
            volt_dict = {f"bmsCellvolt{i + 1}": round(vals[i], 3) for i in range(102)}
            _result = {
                "stepId": id,
                "stepName": stepName_list[i],
                "resultDataList": volt_dict
            }
            result_list.append(_result)

        return result_list
