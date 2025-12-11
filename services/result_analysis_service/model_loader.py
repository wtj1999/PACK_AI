"""
占位模块：如果将来需要加载 ML 模型或外部资源，
把加载/缓存逻辑放这里。示例给出一个简单缓存接口。
"""
import os
import torch
import joblib
from typing import List, Optional
from .models.deepset import DeepSetModel
from .models.packTransformer import PackTransformer


# class ModelHolder:
#     def __init__(self,  settings=None, device='cpu'):
#         self.model_name = settings.MODEL_CONFIG.get('model_name')
#         in_dim = len(settings.MODEL_CONFIG.get('input_feature'))
#         out_dim = len(settings.MODEL_CONFIG.get('out_dim'))
#         node_dim = settings.MODEL_CONFIG.get('node_dim')
#         model_dir_root = settings.MODEL_CONFIG.get('model_store_dir')
#         self.target_idx = '5'
#         self.device = device
#         if self.model_name == 'DeepSet':
#             model = DeepSetModel(in_dim=in_dim,
#                                  emb_dim=settings.MODEL_CONFIG.get(self.model_name)['emb_dim'],
#                                  node_dim=node_dim,
#                                  out_dim=out_dim,
#                                  ).to(device)
#
#         elif self.model_name == 'Transformer':
#             model = PackTransformer(in_dim=in_dim,
#                                     model_dim=settings.MODEL_CONFIG.get(self.model_name)['model_dim'],
#                                     num_layers=settings.MODEL_CONFIG.get(self.model_name)['num_layers'],
#                                     num_heads=settings.MODEL_CONFIG.get(self.model_name)['num_heads'],
#                                     out_dim=out_dim,
#                                     n_cells=node_dim,
#                                     max_rel=51,
#                                     use_pack_token=settings.MODEL_CONFIG.get(self.model_name)['use_pack_token']
#                                     ).to(device)
#         else:
#             raise ValueError(f"Unknown model_name {self.model_name}")
#         self._model = model
#         self.model_dir = os.path.join(model_dir_root, f"{self.model_name}_target{self.target_idx}")
#
#     def load_model(self):
#         self._model.load_state_dict(torch.load(os.path.join(self.model_dir, f"{self.model_name}_best_target{self.target_idx}.pth"),
#                                      map_location=self.device))
#         return self._model, self.model_dir
#
#     def get_model(self):
#         return self._model
#
#     def unload_model(self):
#         self._model = None
class ModelHolder:
    def __init__(self, settings=None, device='cpu', target_idxs: Optional[List[str]] = None):
        """
        settings: 应包含 MODEL_CONFIG 字段
        device: 'cpu' or 'cuda'
        target_idxs: list of target id strings, e.g. ['0','5','6']
        """
        self.model_name = settings.MODEL_CONFIG.get('model_name')
        self.device = device
        self.model_cfg = settings.MODEL_CONFIG
        self.model_dir_root = self.model_cfg.get('model_store_dir')
        self.target_idxs = target_idxs or self.model_cfg.get('target_idxs', ['5'])

    def _build_model_instance(self):

        in_dim = len(self.model_cfg.get('input_feature'))
        out_dim = self.model_cfg.get('out_dim')
        node_dim = self.model_cfg.get('node_dim')

        if self.model_name == 'DeepSet':
            model = DeepSetModel(in_dim=in_dim,
                                 emb_dim=self.model_cfg.get(self.model_name)['emb_dim'],
                                 node_dim=node_dim,
                                 out_dim=out_dim).to(self.device)

        elif self.model_name == 'Transformer':
            model = PackTransformer(in_dim=in_dim,
                                    model_dim=self.model_cfg.get(self.model_name)['model_dim'],
                                    num_layers=self.model_cfg.get(self.model_name)['num_layers'],
                                    num_heads=self.model_cfg.get(self.model_name)['num_heads'],
                                    out_dim=out_dim,
                                    n_cells=node_dim,
                                    max_rel=self.model_cfg.get(self.model_name).get('max_rel', 51),
                                    use_pack_token=self.model_cfg.get(self.model_name).get('use_pack_token', False)
                                    ).to(self.device)
        else:
            raise ValueError(f"Unknown model_name {self.model_name}")
        return model

    def _model_dir_for(self, target_idx: str) -> str:
        return os.path.join(self.model_dir_root, f"{self.model_name}_target{target_idx}")

    def load_model(self, target_idx: str):
        """
        返回 (model_instance, model_dir) —— model 已 load_state_dict 并放到 self.device
        每次调用会创建新的 model instance 并加载对应 target 的权重文件。
        """
        model = self._build_model_instance()
        model_dir = self._model_dir_for(target_idx)
        state_path = os.path.join(model_dir, f"{self.model_name}_best_target{target_idx}.pth")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Model weights not found for target {target_idx}: {state_path}")
        state = torch.load(state_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model, model_dir

    def load_scalers(self, target_idx: str):
        """
        返回 (x_scaler, y_scaler, model_dir)
        """
        model_dir = self._model_dir_for(target_idx)
        x_scaler_path = os.path.join(model_dir, "x_scaler.pkl")
        y_scaler_path = os.path.join(model_dir, "y_scaler.pkl")
        if not os.path.exists(x_scaler_path) or not os.path.exists(y_scaler_path):
            raise FileNotFoundError(f"Scalers not found for target {target_idx} in {model_dir}")
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)
        return x_scaler, y_scaler, model_dir


