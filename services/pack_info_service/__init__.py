from .api import router
from .service import PackinfoService

__all__ = ["router", "PackinfoService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("packinfo", lambda **kw: PackinfoService(**{**service_kwargs, **kw}))