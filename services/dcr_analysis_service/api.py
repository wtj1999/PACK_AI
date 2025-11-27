import logging
from fastapi import APIRouter, Depends, HTTPException
from services.factory import get_service_factory, ServiceFactory
from .schemas import PackQuery, PackDcrResponse
import time
import json

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_factory() -> ServiceFactory:
    return get_service_factory()


@router.post("/pack-dcr-analysis", response_model=PackDcrResponse)
def pack_dcr_analysis(payload: PackQuery, factory: ServiceFactory = Depends(_get_factory)):
    try:
        logger.info("Received /dcr/pack-dcr-analysis request: %s", payload.dict())
    except Exception:
        logger.info("Received /dcr/pack-dcr-analysis request: pack_code=%s",
                    getattr(payload, "pack_code", None))

    try:
        svc = factory.create("dcr")
    except KeyError:
        logger.error("Dcr service not registered; payload=%s", getattr(payload, "dict", lambda: {})())
        raise HTTPException(status_code=500, detail="dcr service 未注册")

    start_pc = time.perf_counter()
    try:
        res = svc.pack_dcr_analysis(payload.pack_code)
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.warning("pack-dcr-analysis ValueError: %s; payload=%s; elapsed_ms=%.2fms",
                       e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-dcr-analysis RuntimeError: %s; payload=%s; elapsed_ms=%.2fms",
                         e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-dcr-analysis unexpected error: %s; payload=%s; elapsed_ms=%.2fms",
                         e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=500, detail=f"内部错误: {e}")

    elapsed_ms = (time.perf_counter() - start_pc) * 1000

    try:
        res_repr = json.dumps(res, ensure_ascii=False, default=str)
    except Exception:
        try:
            res_repr = str(res)
        except Exception:
            res_repr = "<unserializable result>"

    logger.info(
        "pack-dcr-analysis success: pack_code=%s elapsed_ms=%.2fms result=%s",
        payload.pack_code, elapsed_ms, res_repr
    )

    return PackDcrResponse(**res)
