"""
占位模块：如果将来需要加载 ML 模型或外部资源，
把加载/缓存逻辑放这里。示例给出一个简单缓存接口。
"""
# from typing import Optional


class ModelHolder:
    def __init__(self):
        self._model = None

    def load(self, path: str):
        # 实现你的模型加载逻辑
        # 示例: self._model = some_loader(path)
        self._model = {"path": path}
        return self._model

    def get(self):
        return self._model

    def unload(self):
        self._model = None


MODEL_HOLDER = ModelHolder()
