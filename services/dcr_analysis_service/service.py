import pandas as pd
import numpy as np
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from typing import Dict, List, Any, Optional, Tuple
from .util import extract_series_count


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
            outlier_idxs = np.where(z > 5)[0].tolist()

        return outlier_idxs

    def safe_corr(self, a: pd.Series, b: pd.Series) -> Optional[float]:
        mask = a.notna() & b.notna()
        if mask.sum() < 2:
            return None
        if a[mask].std(ddof=0) == 0 or b[mask].std(ddof=0) == 0:
            return None
        return float(a[mask].corr(b[mask]))

    def pack_dcr_analysis(self, pack_codes: List[str]) -> Dict[str, Any]:
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
        dcr_cols = [f'cell_dcr{i + 1}' for i in range(n_cells)]

        rows = []
        for p in pack_codes:
            sub = df[df['pack_code'] == p].copy()
            if not sub.empty and "acquire_time" in sub.columns:
                try:
                    sub = sub.sort_values("acquire_time").reset_index(drop=True)
                except Exception:
                    sub = sub.reset_index(drop=True)

            if sub.empty:
                vals = [np.nan] * n_cells
            else:
                first_row = sub.iloc[0]
                vals = []
                for c in dcr_cols:
                    if c in first_row.index:
                        v = first_row[c]
                        try:
                            fv = float(v) if pd.notna(v) else np.nan
                        except Exception:
                            fv = np.nan
                        vals.append(fv)
                    else:
                        vals.append(np.nan)

            for idx_in_pack, v in enumerate(vals, start=1):
                rows.append({"pack_code": p, "cell_index": int(idx_in_pack), "dcr": float(v) if pd.notna(v) else np.nan})

        dcr_df = pd.DataFrame(rows, columns=["pack_code", "cell_index", "dcr"])

        dcr_list = pd.to_numeric(dcr_df['dcr'], errors='coerce')

        cell_dcr_dict = {
            f"cellDcr{i + 1}": (None if pd.isna(v) else round(float(v), 3))
            for i, v in enumerate(dcr_list)
        }

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

        if len(cell_df) != n_cells * len(pack_codes):
            return {
            "dcr_anomaly_cell_code": [],
            "dcr_list": cell_dcr_dict,
            "correlationAnalysis": []
            }
        if n_cells == 102:
            cell_map_df = pd.read_csv('services/dcr_analysis_service/data/cell_position_map.csv')
        else:
            cell_map_df = pd.read_csv('services/dcr_analysis_service/data/cell_position_map_96.csv')

        dcr_df = pd.merge(dcr_df, cell_map_df, how='left', left_on='cell_index', right_on='cell_index')
        dcr_df = pd.merge(dcr_df, cell_df, how='left', on=['pack_code', 'module_in_pack', 'cell_in_module'])

        outlier_idxs = self._detect_outliers(dcr_df['dcr'].values)
        outlier_df = dcr_df[['pack_code', 'cell_code']].iloc[outlier_idxs]
        records = outlier_df.to_dict(orient="records")

        def _native(x):
            if isinstance(x, (np.generic,)):
                return x.item()
            return x

        records = [{k: _native(v) for k, v in rec.items()} for rec in records]

        corr_dict = {}

        for feat in ['capacity', 'ocv3', 'ocv4', 'acr3', 'acr4', 'k_value', 'cell_thickness', 'weight']:
            feat_list = pd.to_numeric(dcr_df[feat], errors='coerce')
            corr_with = self.safe_corr(feat_list, dcr_list)
            if corr_with:
                corr_with = round(corr_with, 3)
            corr_dict.update({f'corr_with_{feat}': corr_with})

        result = {
            "dcr_anomaly_cell_code": records,
            "dcr_list": cell_dcr_dict,
            "correlationAnalysis": [
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
