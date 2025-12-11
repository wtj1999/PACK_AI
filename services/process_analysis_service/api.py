import logging
from fastapi import APIRouter, Depends, HTTPException
from services.factory import get_service_factory, ServiceFactory
from .schemas import PackQuery, PackProcessResponse
import time
import json

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_factory() -> ServiceFactory:
    return get_service_factory()


@router.post("/pack-process-analysis", response_model=PackProcessResponse)
def pack_process_analysis(payload: PackQuery, factory: ServiceFactory = Depends(_get_factory)):
    try:
        logger.info("Received /process/pack-process-analysis request: %s", payload.dict())
    except Exception:
        logger.info("Received /process/pack-process-analysis request: pack_code=%s",
                    getattr(payload, "pack_code", None))

    try:
        svc = factory.create("process")
    except KeyError:
        logger.error("Process service not registered; payload=%s", getattr(payload, "dict", lambda: {})())
        raise HTTPException(status_code=500, detail="process service 未注册")

    start_pc = time.perf_counter()
    try:
        res = svc.pack_process_analysis(payload.pack_code)
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.warning("pack-process-analysis ValueError: %s; payload=%s; elapsed_ms=%.2fms",
                       e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-process-analysis RuntimeError: %s; payload=%s; elapsed_ms=%.2fms",
                         e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-process-analysis unexpected error: %s; payload=%s; elapsed_ms=%.2fms",
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
        "pack-process-analysis success: pack_code=%s elapsed_ms=%.2fms result=%s",
        payload.pack_code, elapsed_ms, res_repr
    )

    return res
