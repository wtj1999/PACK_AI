from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Dict, Any


class PackQuery(BaseModel):
    pack_code: List = Field(..., description="PACK编号")


class StepResultItem(BaseModel):
    stepId: Optional[Any] = Field(
        None,
        description="步骤 ID（可能为 int/str/float 等），若未知则为 null",
    )
    stepName: Optional[str] = Field(
        None,
        description="步骤名称，例如 '恒流充电'，若未知则为 null",
    )
    resultDataList: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="键为 bmsCellvoltN 的字典，值为对应电压（float），缺失时为 null",
    )
    voltDiff: Optional[float] = Field(
        None,
        description="压差，若未知则为 null",
    )


class PackResultResponse(BaseModel):
    """
    Response model for pack result analysis.
    `results` 是按测试步骤顺序的列表，每项为 StepResultItem。
    """
    results: List[StepResultItem] = Field(
        default_factory=list,
        description="按测试步骤返回的结果列表，每项包含 stepId, stepName 和该步骤的 bmsCellvolt 字典",
    )

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
