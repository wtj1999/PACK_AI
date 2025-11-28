from connects.db_client import DBClient
from fastapi import FastAPI
from services.factory import get_service_factory
from services.temp_corr_service import register as register_temp, router as temp_router
from services.dcr_analysis_service import register as register_dcr, router as dcr_router
from services.result_analysis_service import register as register_result, router as result_router
from core.config import get_settings
from core.logging import setup_logging

setup_logging()
settings = get_settings()
factory = get_service_factory()

service_kwargs = dict()

app = FastAPI(title=settings.APP_NAME)
app.include_router(temp_router, prefix="/temp", tags=["temp"])
app.include_router(dcr_router, prefix="/dcr", tags=["dcr"])
app.include_router(result_router, prefix="/result", tags=["result"])


@app.on_event("startup")
async def startup():
    app.state.db_client = DBClient()
    register_temp(factory, settings=settings, db_client=app.state.db_client, **service_kwargs)
    register_dcr(factory, settings=settings, db_client=app.state.db_client, **service_kwargs)
    register_result(factory, settings=settings, db_client=app.state.db_client, **service_kwargs)

    await factory.startup_all()


@app.on_event("shutdown")
async def shutdown():
    await factory.shutdown_all()
