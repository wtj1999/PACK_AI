from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Dict


class PackQuery(BaseModel):
    pack_code: str = Field(..., description="PACK编号")


class PackResultResponse(BaseModel):
    stepId: str
    stepName: str
    resultDataList: Dict[str, float]

class PredictionItem(BaseModel):
    pred: Optional[List[List[List[float]]]] = Field(
        None,
        description="预测值，若预测失败此字段为null"
    )
    model_dir: Optional[str] = Field(None, description="模型目录")
    error: Optional[str] = Field(None, description="错误信息")

    @root_validator
    def require_pred_or_error(cls, values):
        pred, error = values.get("pred"), values.get("error")
        if pred is None and error is None:
            raise ValueError("PredictionItem 必须至少包含 pred 或 error 之一")
        return values

class PackPredictionsResponse(BaseModel):
    pack_code: str = Field(..., description="PACK编号")
    predictions: Dict[str, PredictionItem] = Field(..., description="预测结果")
    测前压差: Optional[float] = Field(None, description="测前压差")
