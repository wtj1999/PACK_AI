# app/services/temp_service/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class PackQuery(BaseModel):
    vehicle_code: str = Field(..., description="车辆编号")
    step_id: str = Field(..., description="步骤 ID")


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

    corr_minTemp_charge_energy: Optional[float] = Field(
        None, description="minTemp 与 charge_energy 的 Pearson 相关系数（r），不可计算时为 null"
    )
    corr_minTemp_charge_capacity: Optional[float] = Field(
        None, description="minTemp 与 charge_capacity 的 Pearson 相关系数（r），不可计算时为 null"
    )
    corr_minTemp_discharge_energy: Optional[float] = Field(
        None, description="minTemp 与 discharge_energy 的 Pearson 相关系数（r），不可计算时为 null"
    )
    corr_minTemp_discharge_capacity: Optional[float] = Field(
        None, description="minTemp 与 discharge_capacity 的 Pearson 相关系数（r），不可计算时为 null"
    )

    charge_energy_by_time: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="按时间（ISO 字符串）映射的 charge_energy（同时测的 pack 通常相同，取第一个非空），缺失为 null"
    )
    discharge_energy_by_time: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="按时间映射的 discharge_energy，缺失为 null"
    )
    charge_capacity_by_time: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="按时间映射的 charge_capacity，缺失为 null"
    )
    discharge_capacity_by_time: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="按时间映射的 discharge_capacity，缺失为 null"
    )

