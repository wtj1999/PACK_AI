from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, validator

class PackQuery(BaseModel):
    pack_code: List = Field(..., description="PACK编号")

class OutlierItem(BaseModel):
    pack_code: Optional[str] = Field(None, description="pack 的标识（如条码或编号），若缺失则为 null")
    cell_code: Optional[str] = Field(None, description="电芯编码/编号，若缺失则为 null")


class CorrelationItem(BaseModel):
    sourceParam: str = Field(..., description="源参数名称，例如 'DCR'")
    processName: str = Field(..., description="工序/过程名称，例如 'C2500/分容'")
    targetParam: str = Field(..., description="目标参数名称，例如 '电芯总容量' 或 'OCV3'")
    correlationCoefficient: Optional[float] = Field(
        None, description="Pearson 相关系数（四舍五入后的小数）；不可计算时为 null"
    )

class CellDcrItem(BaseModel):
    bmsCellindex: int = Field(..., description="电芯索引，从1开始")
    bmsCelldcr: Optional[float] = Field(
        None, description="电芯dcr（float），缺失时为 null"
    )

class PackDcrResponse(BaseModel):
    """
    返回结构：
      - dcr_anomaly_cell_code: 异常电芯列表（每项包含 pack_code 与 cell_code）
      - dcr_list: 扁平的 cellDcr 序列字典，键如 'cellDcr1'...'cellDcrN'，值为 float 或 null
      - correlationAnalysis: 相关性分析的列表，每项包含 source/process/target 和相关系数
    """
    dcr_anomaly_cell_code: List[OutlierItem] = Field(
        default_factory=list,
        description="检测到的 DCR 异常电芯（list of {pack_code, cell_code}）"
    )
    dcr_list: List[CellDcrItem] = Field(
        default_factory=list,
        description="按顺序的电芯dcr列表，每项包含 bmsCellindex 与 bmsCelldcr。",
    )
    correlationAnalysis: List[CorrelationItem] = Field(
        default_factory=list,
        description="与若干关键工序/参数的相关性分析结果"
    )

