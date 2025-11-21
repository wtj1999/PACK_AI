from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseService(ABC):
    @abstractmethod
    async def startup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self) -> Dict[str, Any]:
        raise NotImplementedError
