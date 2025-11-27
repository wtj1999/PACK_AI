from .api import router
from .service import DcrService

__all__ = ["router", "DcrService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("dcr", lambda **kw: DcrService(**{**service_kwargs, **kw}))
