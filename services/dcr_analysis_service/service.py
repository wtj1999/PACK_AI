from typing import Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from typing import Optional


class DcrService(BaseService):
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
        self.step_id = '1'

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "DcrService", "ready": self._ready}

    def _detect_outliers(self, vals: np.ndarray):

        median = float(np.nanmedian(vals))
        std = float(np.nanstd(vals))

        outlier_idxs = []
        if std > 0:
            z = (vals - median) / std
            outlier_idxs = np.where(z > 4)[0].tolist()

        return outlier_idxs

    def safe_corr(self, a: pd.Series, b: pd.Series) -> Optional[float]:
        mask = a.notna() & b.notna()
        if mask.sum() < 2:
            return None
        if a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
            return None
        return float(a[mask].corr(b[mask]))

    def pack_dcr_analysis(self, pack_code: str) -> Dict[str, Any]:
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        sql = text(f"""
                SELECT *
                FROM `{self.table}`
                WHERE pack_code = :pack_code
                  AND step_id = :step_id
            """)

        try:
            df = self.db_client.read_sql(sql, params={"pack_code": pack_code, "step_id": self.step_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库查询失败: {e}")

        if df.empty:
            raise HTTPException(status_code=404, detail="未查询到任何数据")

        cell_sql = text(f"""
                SELECT pack_code, cell_code, ocv4_time, module_in_pack, cell_in_module, capacity, ocv3, ocv4, acr3, acr4, k_value, cell_thickness, weight
                FROM `{self.table1}`
                WHERE pack_code = :pack_code
            """)

        try:
            cell_df = self.db_client.read_sql(cell_sql, params={"pack_code": pack_code})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"电芯位置数据库查询失败: {e}")

        cell_df = cell_df.sort_values('ocv4_time').drop_duplicates(
            subset=['pack_code', 'cell_code'], keep='last')

        if len(cell_df) != 102:
            raise HTTPException(status_code=404, detail="电芯位置数据读取有误")

        cell_map_df = pd.read_csv('services/dcr_analysis_service/data/cell_position_map.csv')

        volt_cols = [c for c in df.columns if c and c.startswith("cell_dcr")]
        vals = df.iloc[0][volt_cols].values

        dcr_df = pd.DataFrame({
            'cell_index': range(1, len(volt_cols) + 1),
            'cell_dcr': vals
        })

        dcr_df = pd.merge(dcr_df, cell_map_df, how='left', left_on='cell_index', right_on='cell_index')
        dcr_df = pd.merge(dcr_df, cell_df, how='left', on=['module_in_pack', 'cell_in_module'])

        outlier_idxs = self._detect_outliers(vals)
        outlier_cell_code = dcr_df.iloc[outlier_idxs]['cell_code'].tolist()

        dcr_list = pd.to_numeric(dcr_df['cell_dcr'], errors='coerce')

        corr_dict = {}

        for feat in ['capacity', 'ocv3', 'ocv4', 'acr3', 'acr4', 'k_value', 'cell_thickness', 'weight']:
            feat_list = pd.to_numeric(dcr_df[feat], errors='coerce')
            corr_with = self.safe_corr(feat_list, dcr_list)
            if corr_with:
                corr_with = round(corr_with, 3)
            corr_dict.update({f'corr_with_{feat}': corr_with})

        cell_dcr_list = dcr_df.sort_values(['module_in_pack', 'cell_in_module'])['cell_dcr'].values
        cell_dcr_dict = {f"cellDcr{i + 1}": round(cell_dcr_list[i], 3) for i in range(102)}

        result = {
            "dcr_anomaly_cell_code": outlier_cell_code,
            "dcr_list": cell_dcr_dict,
            "correlationAnalysis":[
                {
                    "sourceParam": "DCR",
                    "processName": "C2500/分容",
                    "targetParam": "电芯总容量",
                    "correlationCoefficient": corr_dict['corr_with_capacity'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C2100/二次注液",
                    "targetParam": "后称重量",
                    "correlationCoefficient": corr_dict['corr_with_weight'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C2700/OCV3",
                    "targetParam": "OCV3",
                    "correlationCoefficient": corr_dict['corr_with_ocv3'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C2700/OCV3",
                    "targetParam": "ACR3",
                    "correlationCoefficient": corr_dict['corr_with_acr3'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C2900/OCV4",
                    "targetParam": "OCV4",
                    "correlationCoefficient": corr_dict['corr_with_ocv4'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C2900/OCV4",
                    "targetParam": "ACR4",
                    "correlationCoefficient": corr_dict['corr_with_acr4'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C2900/OCV4",
                    "targetParam": "K值",
                    "correlationCoefficient": corr_dict['corr_with_k_value'],
                },
                {
                    "sourceParam": "DCR",
                    "processName": "C3100/包胶",
                    "targetParam": "电芯厚度",
                    "correlationCoefficient": corr_dict['corr_with_cell_thickness'],
                }
            ]
        }

        return result
