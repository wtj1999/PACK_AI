from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, validator

class PackQuery(BaseModel):
    pack_code: str = Field(..., description="PACK编号")

class CorrelationItem(BaseModel):
    sourceParam: str = Field(..., description="来源参数名")
    processName: str = Field(..., description="过程/工序名称")
    targetParam: str = Field(..., description="目标参数名")
    correlationCoefficient: Optional[float] = Field(None, description="相关系数")

class PackDcrResponse(BaseModel):
    dcr_anomaly_cell_code: List[str]
    dcr_list: Dict[str, float]
    correlationAnalysis: List[CorrelationItem] = Field(
        ..., description="源参数->过程->目标参数的相关性分析结果"
    )

