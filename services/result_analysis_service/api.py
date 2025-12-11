import logging
from fastapi import APIRouter, Depends, HTTPException
from services.factory import get_service_factory, ServiceFactory
from .schemas import PackQuery, PackResultResponse, PackPredictionsResponse
import time
import json
from typing import List

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_factory() -> ServiceFactory:
    return get_service_factory()


@router.post("/pack-result-analysis", response_model=List[PackResultResponse])
def pack_result_analysis(payload: PackQuery, factory: ServiceFactory = Depends(_get_factory)):
    try:
        logger.info("Received /result/pack-result-analysis request: %s", payload.dict())
    except Exception:
        logger.info("Received /result/pack-result-analysis request: pack_code=%s",
                    getattr(payload, "pack_code", None))

    try:
        svc = factory.create("result")
    except KeyError:
        logger.error("Result service not registered; payload=%s", getattr(payload, "dict", lambda: {})())
        raise HTTPException(status_code=500, detail="result service 未注册")

    start_pc = time.perf_counter()
    try:
        res = svc.pack_result_analysis(payload.pack_code)
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.warning("pack-result-analysis ValueError: %s; payload=%s; elapsed_ms=%.2fms",
                       e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-result-analysis RuntimeError: %s; payload=%s; elapsed_ms=%.2fms",
                         e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-result-analysis unexpected error: %s; payload=%s; elapsed_ms=%.2fms",
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
        "pack-result-analysis success: pack_code=%s elapsed_ms=%.2fms result=%s",
        payload.pack_code, elapsed_ms, res_repr
    )

    return res

@router.post("/pack-result-predict", response_model=PackPredictionsResponse)
def pack_result_predict(payload: PackQuery, factory: ServiceFactory = Depends(_get_factory)):
    try:
        logger.info("Received /result/pack-result-predict request: %s", payload.dict())
    except Exception:
        logger.info("Received /result/pack-result-predict request: pack_code=%s",
                    getattr(payload, "pack_code", None))

    try:
        svc = factory.create("result_predict")
    except KeyError:
        logger.error("Result Prediction service not registered; payload=%s", getattr(payload, "dict", lambda: {})())
        raise HTTPException(status_code=500, detail="result predict service 未注册")

    start_pc = time.perf_counter()
    try:
        res = svc.pack_result_predict(payload.pack_code)
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.warning("pack-result-predict ValueError: %s; payload=%s; elapsed_ms=%.2fms",
                       e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-result-predict RuntimeError: %s; payload=%s; elapsed_ms=%.2fms",
                         e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-result-predict unexpected error: %s; payload=%s; elapsed_ms=%.2fms",
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
        "pack-result-predict success: pack_code=%s elapsed_ms=%.2fms result=%s",
        payload.pack_code, elapsed_ms, res_repr
    )

    return PackPredictionsResponse(**res)

