from .api import router
from .service import ResultService

__all__ = ["router", "ResultService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("result", lambda **kw: ResultService(**{**service_kwargs, **kw}))
