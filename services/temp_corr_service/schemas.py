# app/services/temp_service/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class PackQuery(BaseModel):
    vehicle_code: str = Field(..., description="车辆编号")
    step_id: str = Field(..., description="步骤 ID")

class ChargeEnergyItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    charge_energy: Optional[float] = Field(None, description="按时间取 charge_energy（第一个非空）")

class DischargeEnergyItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    discharge_energy: Optional[float] = Field(None, description="按时间取 discharge_energy（第一个非空）")

class ChargeCapacityItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    charge_capacity: Optional[float] = Field(None, description="按时间取 charge_capacity（第一个非空）")

class DischargeCapacityItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    discharge_capacity: Optional[float] = Field(None, description="按时间取 discharge_capacity（第一个非空）")

class TempItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    temp_min: Optional[float] = Field(None, description="按时间取 temp_min（第一个非空）")

class PackCorrResponse(BaseModel):
    """
    Response model for pack_temp_corr results.
    Time-keyed dicts use ISO-format acquire_time strings as keys.
    Correlations are Pearson r (float) or null when not computable.
    """
    vehicle_code: Optional[str] = Field(
        None, description="原始 vehicle_code（如果从 DB 读取到则返回，否则为 null）"
    )
    step_id: Optional[str] = Field(
        None, description="请求时传入的 step_id（或 'all'）"
    )

    corr_minTemp_energy: Optional[float] = Field(
        None, description="minTemp 与 energy 的 Pearson 相关系数（r），不可计算时为 null"
    )
    corr_minTemp_capacity: Optional[float] = Field(
        None, description="minTemp 与 capacity 的 Pearson 相关系数（r），不可计算时为 null"
    )

    temp_min_list: List[TempItem] = Field(
        default_factory=list,
        description="按时间的 temp_min 列表（第一个非空），按时间升序"
    )
    charge_energy_list: List[ChargeEnergyItem] = Field(
        default_factory=list,
        description="按时间的 charge_energy 列表（第一个非空），按时间升序"
    )
    discharge_energy_list: List[DischargeEnergyItem] = Field(
        default_factory=list,
        description="按时间的 discharge_energy 列表（第一个非空），按时间升序"
    )
    charge_capacity_list: List[ChargeCapacityItem] = Field(
        default_factory=list,
        description="按时间的 charge_capacity 列表（第一个非空），按时间升序"
    )
    discharge_capacity_list: List[DischargeCapacityItem] = Field(
        default_factory=list,
        description="按时间的 discharge_capacity 列表（第一个非空），按时间升序"
    )

