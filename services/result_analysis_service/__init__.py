from .api import router
from .service import ResultService, ResultPredictService

__all__ = ["router", "ResultService"]


def register(factory, settings=None, **service_kwargs):

    factory.register("result", lambda **kw: ResultService(**{**service_kwargs, **kw}))
    factory.register(
        "result_predict",
        lambda **kw: ResultPredictService(settings=settings, **{**service_kwargs, **kw})
    )
