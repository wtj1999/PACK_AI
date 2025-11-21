# app/services/temp_service/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional


class PackQuery(BaseModel):
    vehicle_code: str = Field(..., description="车辆编号")
    step_id: str = Field(..., description="步骤 ID")


class PackCorrResponse(BaseModel):
    vehicle_code: str
    step_id: str
    step_name: Optional[str]
    corr_minTemp_energy: Optional[float]
    corr_minTemp_capacity: Optional[float]
    min_temp_list: List[Optional[float]]
    energy_list: List[Optional[float]]
    capacity_list: List[Optional[float]]
