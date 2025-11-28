from pydantic import BaseModel, Field
from typing import Dict


class PackQuery(BaseModel):
    pack_code: str = Field(..., description="PACK编号")


class PackResultResponse(BaseModel):
    stepId: str
    stepName: str
    resultDataList: Dict[str, float]
