from .api import router
from .service import TempService

__all__ = ["router", "TempService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("temp", lambda **kw: TempService(**{**service_kwargs, **kw}))
