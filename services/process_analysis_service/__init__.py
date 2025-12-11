from .api import router
from .service import ProcessService

__all__ = ["router", "ProcessService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("process", lambda **kw: ProcessService(**{**service_kwargs, **kw}))
