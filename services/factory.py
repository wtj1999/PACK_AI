# app/s
from __future__ import annotations
import inspect
import asyncio
from functools import lru_cache
from typing import Callable, Dict, Optional, Any, List

from .base import BaseService
import logging

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    简单的 ServiceFactory：负责注册 service 构造器并
    提供创建/获取实例、统一启动/关闭等功能。

    用法示例:
        factory = get_service_factory()
        factory.register("temp", TempService)         # 注册类（延迟实例化）
        factory.register("pack", lambda **kw: PackService(**kw))  # 注册工厂函数
        svc = factory.create("temp", model_path="...")  # 创建/获取实例
    """

    def __init__(self):
        self._registry: Dict[str, Callable[..., BaseService]] = {}
        self._instances: Dict[str, BaseService] = {}

    def register(self, name: str, ctor: Callable[..., BaseService]) -> None:
        if not callable(ctor):
            raise TypeError("ctor must be callable")
        logger.debug("Register service %s -> %s", name, getattr(ctor, "__name__", str(ctor)))
        self._registry[name] = ctor

    def unregister(self, name: str) -> None:
        """注销注册并删除已实例"""
        logger.debug("Unregister service %s", name)
        self._registry.pop(name, None)
        # inst = self._instances.pop(name, None)

    def create(self, name: str, *, force_new: bool = False, **kwargs: Any) -> BaseService:
        """
        创建或返回已存在实例。
        - name: 注册时使用的 key
        - force_new: 若 True 每次强制创建新实例（并覆盖旧实例）
        - kwargs: 传给构造器的参数
        """
        if name not in self._registry:
            raise KeyError(f"service '{name}' is not registered")

        if not force_new and name in self._instances:
            return self._instances[name]

        ctor = self._registry[name]
        if inspect.isclass(ctor):
            inst = ctor(**kwargs)
        else:
            inst = ctor(**kwargs)

        if not isinstance(inst, BaseService):
            raise TypeError("created object is not an instance of BaseService")

        self._instances[name] = inst
        logger.info("Service '%s' instantiated", name)
        return inst

    def get(self, name: str) -> Optional[BaseService]:
        return self._instances.get(name)

    def list_registered(self) -> List[str]:
        return list(self._registry.keys())

    def list_instances(self) -> List[str]:
        return list(self._instances.keys())

    def clear_instances(self) -> None:
        self._instances.clear()

    # ---------------- lifecycle ----------------
    async def startup_all(self) -> None:
        """
        依次对所有注册服务实例化（若尚未实例化则创建），并调用其 startup 方法（并行）。
        注意：若某些 service 的 constructor 需要耗时初始化，建议在 constructor 里不做重操作，
        把加载放到 startup()。
        """
        for name in list(self._registry.keys()):
            if name not in self._instances:
                try:
                    self.create(name)
                except Exception:
                    logger.exception("failed to instantiate service %s during startup_all", name)

        coros = []
        for name, inst in self._instances.items():
            if hasattr(inst, "startup"):
                try:
                    coro = inst.startup()
                    if asyncio.iscoroutine(coro):
                        coros.append(coro)
                except Exception:
                    logger.exception("service %s startup() raised synchronously", name)

        if coros:
            await asyncio.gather(*coros, return_exceptions=False)
        logger.info("ServiceFactory: startup_all finished for %s", ", ".join(self._instances.keys()))

    async def shutdown_all(self) -> None:
        """
        依次调用已创建实例的 shutdown（并行）。调用后不自动删除实例，若需要清理可再调用 clear_instances().
        """
        coros = []
        for name, inst in self._instances.items():
            if hasattr(inst, "shutdown"):
                try:
                    coro = inst.shutdown()
                    if asyncio.iscoroutine(coro):
                        coros.append(coro)
                except Exception:
                    logger.exception("service %s shutdown() raised synchronously", name)

        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        logger.info("ServiceFactory: shutdown_all finished for %s", ", ".join(self._instances.keys()))

    def info_all(self) -> Dict[str, Dict[str, Any]]:
        """返回所有已实例化服务的 info() 字典集合"""
        out: Dict[str, Dict[str, Any]] = {}
        for name, inst in self._instances.items():
            try:
                out[name] = inst.info()
            except Exception:
                logger.exception("service %s info() failed", name)
                out[name] = {"error": True}
        return out


@lru_cache()
def get_service_factory() -> ServiceFactory:
    return ServiceFactory()
