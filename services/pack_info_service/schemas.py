from pydantic import BaseModel, Field, root_validator
from typing import List, Optional, Dict, Any


class PackQuery(BaseModel):
    vehicleCode: Optional[str] = Field(None, description="车辆编码 (device_code)")
    startTime: Optional[str] = Field(None, description="开始时间，格式示例 'YYYY-MM-DD HH:MM:SS'")
    endTime: Optional[str] = Field(None, description="结束时间，格式示例 'YYYY-MM-DD HH:MM:SS'")
    elecProcessConfig: Optional[str] = Field(None, description="电测配置")

class PackInfoItem(BaseModel):
    vehicleCode: str = Field(..., description="车辆码")
    elecProcessConfig: Optional[Any] = Field(
        None,
        description="电气工艺配置（如 '330阶梯充一拖四/三'），若未知为 null"
    )
    packCodeList: List[str] = Field(
        default_factory=list,
        description="去重后的 pack_code 列表，字符串形式"
    )
    dateTime: Optional[str] = Field(
        None,
        description="时间字符串，格式例如 '2026-02-01 00:00:00'，若未知为 null"
    )

class PackInfoResponse(BaseModel):
    """
    返回的聚合结果：按 vehicle 分组的 pack 列表与元信息
    """
    results: List[PackInfoItem] = Field(
        default_factory=list,
        description="按 vehicle 聚合的结果列表，每项为 PackInfoItem"
    )