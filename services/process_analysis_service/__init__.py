from .api import router
from .service import ProcessService, ProcessDisplayService

__all__ = ["router", "ProcessService", "ProcessDisplayService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("process", lambda **kw: ProcessService(**{**service_kwargs, **kw}))
    factory.register("process_display", lambda **kw: ProcessDisplayService(settings=settings, **{**service_kwargs, **kw}))
