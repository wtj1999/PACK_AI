import logging
import torch
import re
import pandas as pd
import numpy as np
from sqlalchemy import text
from services.base import BaseService
from fastapi import HTTPException
from .util import DataValidator, PackFrameBuilder, build_pack_features
from .model_loader import ModelHolder
from typing import Dict, List, Any, Optional, Tuple
import difflib


logger = logging.getLogger(__name__)

class ResultService(BaseService):
    """
    TempService: 提供 pack-temp-corr 功能的 service 类。
    推荐在注册时把 engine 与列名通过构造器注入，例如：
        factory.register("temp", lambda **kw: TempService(engine=engine, table='your_table', temp_cols_per_pack=[...]))
    """
    def __init__(self, settings=None, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.table = 'jz2_pack_result_data'
        self.test_step_map = settings.TEST_STEP_CONFIG
        self.volt_cols_per_pack = [f"bms_cellvolt{i}" for i in range(1, 103)]

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "ResultService", "ready": self._ready}

    def normalize_label(self, s: str) -> str:
        COMMON_NOISE_PATTERNS = [
            r"\b测试\b",  # 测试
            r"\bDCR\b",  # DCR
            r"\b\d+P\d+S\b",  # 1P102S, etc.
            r"\b\d+P\b",  # 1P
            r"\bP\d+S\b",  # P102S
            r"\b1P102S\b",  # explicit
            r"[()（）\-_/]",  # 括号和连接符
            r"\s+",  # 多余空白
        ]

        _noise_re = re.compile("|".join(COMMON_NOISE_PATTERNS), flags=re.IGNORECASE)

        if s is None:
            return ""
        s = str(s)
        s = _noise_re.sub(" ", s)
        s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", s)
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    def find_best_test_config_key(self,
                                  label: str,
                                  config: Dict[str, Dict],
                                  fuzzy_threshold: float = 0.8
                                  ) -> Optional[Tuple[str, Dict]]:
        if not label:
            return None

        label_norm = self.normalize_label(label)
        norm_map = {}
        for k in config.keys():
            kn = self.normalize_label(k)
            norm_map[k] = kn

        for orig_k, kn in norm_map.items():
            if kn and kn in label_norm:
                return orig_k, config[orig_k]

        for orig_k, kn in norm_map.items():
            if label_norm and label_norm in kn:
                return orig_k, config[orig_k]

        best_k = None
        best_score = 0.0
        for orig_k, kn in norm_map.items():
            if not kn:
                continue
            score = difflib.SequenceMatcher(None, label_norm, kn).ratio()
            if score > best_score:
                best_score = score
                best_k = orig_k

        if best_k and best_score >= fuzzy_threshold:
            return best_k, config[best_k]

        return None

    def pack_result_analysis(self, pack_codes: List[str]) -> dict[str, List]:
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

        pack_dfs: Dict[str, pd.DataFrame] = {}
        charge_energy_list = []
        discharge_energy_list = []
        charge_capacity_list = []
        discharge_capacity_list = []
        for p in pack_codes:
            sub = df[df['pack_code'] == p].copy()
            sub = sub.sort_values('acquire_time').reset_index(drop=True)
            charge_energy_list.append(sub['charge_energy'].values)
            charge_capacity_list.append(sub['charge_capacity'].values)
            discharge_energy_list.append(sub['discharge_energy'].values)
            discharge_capacity_list.append(sub['discharge_capacity'].values)
            pack_dfs[p] = sub

        lengths = [len(g) for g in pack_dfs.values()]
        min_len = min(lengths) if lengths else 0
        if min_len == 0:
            raise HTTPException(status_code=500, detail="至少有一个 pack 没有结果数据")

        volt_concat_list = []
        for idx, p in enumerate(pack_codes):
            g = pack_dfs[p].iloc[:min_len].reset_index(drop=True)

            offset = idx * len(self.volt_cols_per_pack)
            rename_map = {old: f'BMS_BattVolt{offset + i + 1}' for i, old in enumerate(self.volt_cols_per_pack)}
            tdf_renamed = g[self.volt_cols_per_pack].rename(columns=rename_map).reset_index(drop=True)
            volt_concat_list.append(tdf_renamed)

        meta_frames = g[['vehicle_code', 'step_id', 'acquire_time']].reset_index(drop=True)

        volt_df = pd.concat(volt_concat_list, axis=1)
        volt_df = pd.concat([meta_frames, volt_df], axis=1)

        volt_cols = [col for col in volt_df.columns if col and col.startswith("BMS_BattVolt")]

        res = self.find_best_test_config_key(df['vehicle_to_pack_num'].iloc[0], self.test_step_map)
        test_step_info = res[1] if res else None

        if not test_step_info:
            raise HTTPException(status_code=500, detail="测试步骤信息未配置")

        result_list = []

        for step_id, step_name in test_step_info.items():
            if volt_df[volt_df['step_id'] == step_id].empty:
                _result = {
                    "stepId": step_id,
                    "stepName": step_name,
                    "resultDataList": {f"bmsCellvolt{i + 1}": None for i in range(len(volt_cols))},
                    "voltDiff": None
                }
            else:
                volt_data = volt_df[volt_df['step_id'] == step_id][volt_cols].iloc[0].values
                volt_dict = {
                            f"bmsCellvolt{i + 1}": (None if pd.isna(v) else round(float(v), 3))
                            for i, v in enumerate(volt_data)
                            }
                volt_list = [
                    {
                        'bmsCellindex': i + 1,
                        'bmsCellvolt': (None if pd.isna(v) else round(float(v), 3))
                    }
                    for i, v in enumerate(volt_data)
                ]
                _result = {
                    "stepId": step_id,
                    "stepName": step_name,
                    "resultDataList": volt_list,
                    "voltDiff": round(np.max(volt_data[volt_data != None]) - np.min(volt_data[volt_data != None]), 3)
                }
            result_list.append(_result)

        return {'results': result_list}

class ResultPredictService(BaseService):

    def __init__(self, settings=None, db_client=None):
        self.db_client = db_client
        self._ready = False
        self.frame_builder = PackFrameBuilder()
        self.cell_feature = ['capacity', 'ocv3', 'ocv4', 'acr3', 'acr4', 'k_value', 'cell_thickness', 'weight']
        self.test_step_feature = ['1']
        self.target_name = ['Discharge_Dynamic_Voltage', 'Discharge_Static_Voltage', 'Charge_Dynamic_Voltage', 'Charge_Static_Voltage']
        self.result_table = 'jz2_pack_result_data'
        self.cell_table = 'jz2_pack_cell_data'
        self.model_name = 'Catboost'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_holder = ModelHolder(settings=settings, device=self.device, target_name=self.target_name)

    async def startup(self) -> None:
        self._ready = True

    async def shutdown(self) -> None:
        self._ready = False

    def info(self) -> Dict[str, Any]:
        return {"name": "ResultPredictService", "ready": self._ready}

    def pack_result_predict(self, pack_codes: List[str]):
        input_df = self.fetch_input_data(pack_codes)
        pred_result = {}

        if self.model_name == 'Catboost':

            input_tree_df = build_pack_features(
                input_df,
                group_col='vehicle_code',
                numeric_cols=self.cell_feature,
                step_range_for_inputs= self.test_step_feature,
                stats=['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range'],
                include_counts=True
            )

            numeric_cols = input_tree_df.select_dtypes(include=[np.number]).columns.tolist()
            for target in self.target_name:
                try:
                    model, model_dir = self.model_holder.load_model(target)
                except Exception as e:
                    pred_result.update({
                        f"{target}": 0
                    })
                    continue

                y_pred = model.predict(input_tree_df[numeric_cols])
                pred_result.update({
                    f"{target}": round(y_pred[0], 3)
                })

        else:
            input_vals = input_df.values

            if input_vals.shape != (self.cell_num, self.input_feature_num):
                raise HTTPException(status_code=500,
                                    detail=f"input shape mismatch, got {input_vals.shape}, expected ({self.cell_num},{self.input_feature_num})")

            X_flat = input_vals.reshape(-1, self.input_feature_num)  # (cell_num, feature)

            results = {"pack_code": pack_codes, "predictions": {}}

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

        return pred_result


    def fetch_cell_data(self, pack_codes: List[str]) -> pd.DataFrame:
        sql = text(f"""
                            SELECT *
                            FROM `{self.cell_table}`
                            WHERE pack_code IN :pack_codes
                            """)
        try:
            df = self.db_client.read_sql(sql, params={"pack_codes": pack_codes})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库电芯数据查询失败: {e}")

        df = df.sort_values('ocv4_time').drop_duplicates(
            subset=['pack_code', 'module_code', 'cell_code'], keep='last')
        df = df.query("cell_code.notna() and cell_code != ''")

        if not DataValidator.is_valid_pack_cell_df(df):
            raise HTTPException(status_code=404, detail="数据库中电芯数据不完整")

        return df

    def fetch_result_data(self, pack_codes: List[str]) -> pd.DataFrame:
        sql = text(f"""
                    SELECT *
                    FROM `{self.result_table}`
                    WHERE pack_code IN :pack_codes
                    AND step_id IN :step_ids
                    """)
        try:
            df = self.db_client.read_sql(sql, params={"pack_codes": pack_codes, "step_ids": self.test_step_feature})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"数据库结果数据查询失败: {e}")

        df = df.sort_values(["acquire_time"]).drop_duplicates(subset=['pack_code', 'step_id'], keep='last')

        if not DataValidator.is_valid_pack_result_df(df, self.test_step_feature):
            raise HTTPException(status_code=404, detail="数据库中结果数据不完整")

        return df

    def fetch_input_data(self, pack_codes: List[str]):

        cell_df = self.fetch_cell_data(pack_codes)
        result_df = self.fetch_result_data(pack_codes)

        pack_codes, flag = DataValidator.is_valid_cell_result_df(result_df, cell_df)

        if not flag:
            raise HTTPException(status_code=404, detail="当前电测数据和电芯数据无法匹配")

        n_cells = DataValidator.extract_series_count(result_df.iloc[0]['vehicle_to_pack_num'])

        if n_cells != 102 and n_cells != 96:
            raise HTTPException(status_code=404, detail="电芯数目解析错误")

        cell_map_df = pd.read_csv('services/result_analysis_service/data/cell_position_map.csv') if n_cells == 102 else pd.read_csv(
            'services/result_analysis_service/data/cell_position_map_96.csv')

        cell_df = pd.merge(cell_df,
                           cell_map_df,
                           on=['module_in_pack', 'cell_in_module'],
                           how='left')

        df = self.frame_builder.build_frames(result_df, cell_df, pack_codes, n_cells)
        return df






