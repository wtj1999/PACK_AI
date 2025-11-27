from pydantic import BaseModel, Field
from typing import List, Dict


class PackQuery(BaseModel):
    pack_code: str = Field(..., description="PACK编号")


class PackDcrResponse(BaseModel):
    dcr_anomaly_cell_code: List[str]
    dcr_list: Dict[str, float]
