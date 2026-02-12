from .api import router
from .service import CellinfoService

__all__ = ["router", "CellinfoService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("cellinfo", lambda **kw: CellinfoService(**{**service_kwargs, **kw}))