import logging
from fastapi import APIRouter, Depends, HTTPException
from services.factory import get_service_factory, ServiceFactory
from .schemas import PackQuery, PackCorrResponse
import time
import json

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_factory() -> ServiceFactory:
    return get_service_factory()

# @router.post("/pack-temp-corr", response_model=PackCorrResponse)
# def pack_temp_corr(payload: PackQuery, factory: ServiceFactory = Depends(_get_factory)):
#     """
#     Router 层：通过 factory 获取/创建 TempService 并执行 pack_temp_corr。
#     注意：factory.create 返回的是 TempService 实例（请在 main.py 中用 register 注册）。
#     """
#     try:
#         logger.info("Received /temp/pack-temp-corr request: %s", payload.dict())
#     except Exception:
#         logger.info("Received /temp/pack-temp-corr request: vehicle_code=%s step_id=%s",
#                     getattr(payload, "vehicle_code", None), getattr(payload, "step_id", None))
#
#     try:
#         svc = factory.create("temp")
#     except KeyError:
#         raise HTTPException(status_code=500, detail="temp service 未注册")
#
#     try:
#         res = svc.pack_temp_corr(payload.vehicle_code, payload.step_id)
#     except ValueError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except RuntimeError as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"内部错误: {e}")
#
#     logger.info("pack-temp-corr success: vehicle_code=%s step_id=%s", payload.vehicle_code, payload.step_id)
#
#     return PackCorrResponse(**res)


@router.post("/pack-temp-corr", response_model=PackCorrResponse)
def pack_temp_corr(payload: PackQuery, factory: ServiceFactory = Depends(_get_factory)):
    try:
        logger.info("Received /temp/pack-temp-corr request: %s", payload.dict())
    except Exception:
        logger.info("Received /temp/pack-temp-corr request: vehicle_code=%s step_id=%s",
                    getattr(payload, "vehicle_code", None), getattr(payload, "step_id", None))

    try:
        svc = factory.create("temp")
    except KeyError:
        logger.error("Temp service not registered; payload=%s", getattr(payload, "dict", lambda: {})())
        raise HTTPException(status_code=500, detail="temp service 未注册")

    start_pc = time.perf_counter()
    try:
        res = svc.pack_temp_corr(payload.vehicle_code, payload.step_id)
    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.warning("pack-temp-corr ValueError: %s; payload=%s; elapsed_ms=%.2fms",
                       e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-temp-corr RuntimeError: %s; payload=%s; elapsed_ms=%.2fms",
                         e, getattr(payload, "dict", lambda: {})(), elapsed_ms)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_pc) * 1000
        logger.exception("pack-temp-corr unexpected error: %s; payload=%s; elapsed_ms=%.2fms",
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
        "pack-temp-corr success: vehicle_code=%s step_id=%s elapsed_ms=%.2fms result=%s",
        payload.vehicle_code, payload.step_id, elapsed_ms, res_repr
    )

    return PackCorrResponse(**res)
