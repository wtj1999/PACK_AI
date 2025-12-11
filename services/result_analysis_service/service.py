import logging
import torch
import os
import joblib
import torch.nn as nn
import pandas as pd
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from .util import DataValidator, PackFrameBuilder
from .model_loader import ModelHolder
from typing import Dict, List, Any


logger = logging.getLogger(__name__)

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


class ResultPredictService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, settings=None, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.result_table = 'jz2_pack_result_data'
        self.cell_table = 'jz2_pack_cell_data'
        self.cell_map_df = pd.read_csv('services/result_analysis_service/data/cell_position_map.csv')
        self.frame_builder = PackFrameBuilder()
        self.input_feature = settings.MODEL_CONFIG.get('input_feature')
        self.input_feature_num = len(self.input_feature)
        self.cell_num = settings.PACK_CONFIG.get('CELLS_PER_PHYSICAL_PACK')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_idxs = settings.MODEL_CONFIG.get('target_idxs', ['0', '5', '6'])
        self.model_holder = ModelHolder(settings=settings, device=self.device, target_idxs=self.target_idxs)


    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "ResultPredictService", "ready": self._ready}

    def fetch_cell_data(self, pack_code: str) -> pd.DataFrame:

        sql = text(f"""
                    SELECT *
                    FROM `{self.cell_table}`
                    WHERE pack_code = :pack_code
                    """)
        try:
            df = self.db_client.read_sql(sql, params={"pack_code": pack_code})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库电芯数据查询失败: {e}")

        df = df.sort_values('ocv4_time').drop_duplicates(
            subset=['pack_code', 'module_code', 'cell_code'], keep='last')

        if not DataValidator.is_valid_pack_cell_df(df):
            raise HTTPException(status_code=404, detail="数据库中电芯数据不完整")

        return df

    def fetch_result_data(self, pack_code: str) -> pd.DataFrame:
        sql = text(f"""
                    SELECT *
                    FROM `{self.result_table}`
                    WHERE pack_code = :pack_code
                    """)
        try:
            df = self.db_client.read_sql(sql, params={"pack_code": pack_code})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库结果数据查询失败: {e}")

        df = df.sort_values(["acquire_time"]).drop_duplicates(subset=['step_id'], keep='last')

        if not DataValidator.is_valid_pack_result_df(df):
            raise HTTPException(status_code=404, detail="数据库中结果数据不完整")

        return df

    def fetch_input_data(self, pack_code: str):
        if self.db_client is None:
            raise HTTPException(status_code=500, detail="数据库引擎创建失败")

        cell_df = self.fetch_cell_data(pack_code)
        result_df = self.fetch_result_data(pack_code)

        cell_df = pd.merge(cell_df,
                           self.cell_map_df,
                           on=['module_in_pack', 'cell_in_module'],
                           how='inner')

        df = self.frame_builder.build_frames(cell_df, result_df)
        input_df = df[self.input_feature]
        if input_df.isnull().any().any():
            raise HTTPException(status_code=404, detail="数据库中数据不完整,不进行预测任务")

        return input_df

    def pack_result_predict(self, pack_code: str):

        input_df = self.fetch_input_data(pack_code)
        input_vals = input_df.values

        if input_vals.shape != (self.cell_num, self.input_feature_num):
            raise HTTPException(status_code=500,
                                detail=f"input shape mismatch, got {input_vals.shape}, expected ({self.cell_num},{self.input_feature_num})")

        X_flat = input_vals.reshape(-1, self.input_feature_num)  # (cell_num, feature)

        results = {"pack_code": pack_code, "predictions": {}}

        for target in self.target_idxs:
            try:
                model, model_dir = self.model_holder.load_model(target)
                x_scaler, y_scaler, _ = self.model_holder.load_scalers(target)
            except Exception as e:
                results["predictions"][target] = {"error": str(e)}
                continue

            X_scaled_flat = x_scaler.transform(X_flat)  # (cell_num, feature)
            X_scaled = X_scaled_flat.reshape(1, self.cell_num, self.input_feature_num)
            X_tensor = torch.from_numpy(X_scaled).float().to(self.device)

            with torch.no_grad():
                model.to(self.device)
                model.eval()
                pred_np = model(X_tensor)  # (1, cell_num, out_dim)
                if isinstance(pred_np, torch.Tensor):
                    pred_np = pred_np.cpu().numpy()

            out_dim = pred_np.shape[-1]
            pred_flat = pred_np.reshape(-1, out_dim)
            try:
                pred_inv_flat = y_scaler.inverse_transform(pred_flat)
            except Exception:
                if pred_flat.shape[1] == 1:
                    pred_inv_flat = y_scaler.inverse_transform(pred_flat)
                else:
                    pred_inv_flat = pred_flat

            pred_final = pred_inv_flat.reshape(1, self.cell_num, -1)  # (1, cell_num, out_dim)
            pred_list = pred_final.tolist()

            results["predictions"][target] = {
                "pred": pred_list,
                "model_dir": model_dir
            }

        return results





