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

        # charge_energy_list = np.array(charge_energy_list)
        # charge_capacity_list = np.array(charge_capacity_list)
        # discharge_energy_list = np.array(discharge_energy_list)
        # discharge_capacity_list = np.array(discharge_capacity_list)

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
                    "resultDataList": {f"bmsCellvolt{i + 1}": 0.0 for i in range(len(volt_cols))}
                }
            else:
                volt_data = volt_df[volt_df['step_id'] == step_id][volt_cols].iloc[0].values
                volt_dict = {f"bmsCellvolt{i + 1}": round(volt_data[i], 3) for i in range(len(volt_data))}
                _result = {
                    "stepId": step_id,
                    "stepName": step_name,
                    "resultDataList": volt_dict,
                    "voltDiff": round(np.max(volt_data) - np.min(volt_data), 3)
                }
            result_list.append(_result)
        # if len(df) < 5:
        #     raise HTTPException(status_code=404, detail="数据库中结果数据不完整")

        # volt_cols = [c for c in df.columns if c and c.startswith("bms_cellvolt")]
        # stepName_list = ['测前电压', '充电末端动态电压', '充电后静态电压', '放电末端动态电压', '放电后静态电压']
        # cell_map_df = pd.read_csv('services/result_analysis_service/data/cell_position_map.csv')
        # result_list = []
        #
        # for i, id in enumerate(['1', '8', '9', '14', '15']):
        #     if df[df['step_id'] == id].empty:
        #         _result = {
        #             "stepId": id,
        #             "stepName": stepName_list[i],
        #             "resultDataList": {f"bmsCellvolt{i + 1}": 0.0 for i in range(102)}
        #         }
        #     else:
        #         volt_data = df[df['step_id'] == id][volt_cols].iloc[0].values
        #         volt_df = pd.DataFrame({
        #             'cell_index': range(1, len(volt_cols) + 1),
        #             'cell_volt': volt_data
        #         })
        #         volt_df = pd.merge(volt_df, cell_map_df, how='left', left_on='cell_index', right_on='cell_index')
        #         vals = volt_df.sort_values(['module_in_pack', 'cell_in_module'])['cell_volt'].values
        #         volt_dict = {f"bmsCellvolt{i + 1}": round(vals[i], 3) for i in range(102)}
        #         _result = {
        #             "stepId": id,
        #             "stepName": stepName_list[i],
        #             "resultDataList": volt_dict
        #         }
        #     result_list.append(_result)

        return {'results': result_list}


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
        self.target_name = settings.MODEL_CONFIG.get('target_name', ['Discharge_Dynamic_Voltage', 'Discharge_Static_Voltage'])
        self.model_name = settings.MODEL_CONFIG.get('model_name', 'Catboost')
        self.model_holder = ModelHolder(settings=settings, device=self.device, target_name=self.target_name)


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
        return df

    def cal_diff_data(self, df):
        diff_result = {}
        if 'step_1_volt' in df.columns:
            diff_result.update({
                "测前压差": round(df['step_1_volt'].max() - df['step_1_volt'].min(), 3)
            })
        else:
            diff_result.update({
                "测前压差": 0
            })
        if 'step_8_volt' in df.columns:
            diff_result.update({
                "充电末端动态压差": round(df['step_8_volt'].max() - df['step_8_volt'].min(), 3)
            })
        else:
            diff_result.update({
                "充电末端静态压差": 0
            })
        if 'step_9_volt' in df.columns:
            diff_result.update({
                "充电后静态压差": round(df['step_9_volt'].max() - df['step_9_volt'].min(), 3)
            })
        else:
            diff_result.update({
                "充电后静态压差": 0
            })
        if 'step_14_volt' in df.columns:
            diff_result.update({
                "放电末端动态压差": round(df['step_14_volt'].max() - df['step_14_volt'].min(), 3)
            })
        else:
            diff_result.update({
                "放电末端动态压差": 0
            })
        if 'step_15_volt' in df.columns:
            diff_result.update({
                "放电后静态压差": round(df['step_15_volt'].max() - df['step_15_volt'].min(), 3)
            })
        else:
            diff_result.update({
                "放电后静态压差": 0
            })

        return diff_result

    def pack_result_predict(self, pack_code: str):

        input_df = self.fetch_input_data(pack_code)

        diff_result = self.cal_diff_data(input_df)

        if input_df[self.input_feature].isnull().any().any():
            diff_result.update({
                "放电末端动态压差预测值": 0,
                "放电后静态压差预测值": 0
            })
            return diff_result

        if self.model_name == 'Catboost':
            target_name_map = {
                'Discharge_Dynamic_Voltage': '放电末端动态压差预测值',
                'Discharge_Static_Voltage': '放电后静态压差预测值'
            }
            input_tree_df = build_pack_features(
                input_df,
                group_col='pack_code',
                numeric_cols=None,
                step_range_for_inputs=range(1, 10),
                stats=['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range'],
                include_counts=True
            )
            numeric_cols = input_tree_df.select_dtypes(include=[np.number]).columns.tolist()
            for target in self.target_name:
                try:
                    model, model_dir = self.model_holder.load_model(target)
                except Exception as e:
                    diff_result.update({
                        f"{target_name_map[target]}": 0
                    })
                    continue

                y_pred = model.predict(input_tree_df[numeric_cols])
                diff_result.update({
                    f"{target_name_map[target]}": round(y_pred[0], 3)
                })

        else:
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

        return diff_result





