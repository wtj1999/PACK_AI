# models.py
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class PackQuery(BaseModel):
    pack_code: List = Field(..., description="PACK编号")

class AnalysisResult(BaseModel):
    """核心分析指标"""
    over_ratio: float = Field(..., description="越界点比例（点数比例）")
    max_continuous_over_s: float = Field(..., description="最长连续越界时长（秒）")
    dtw_distance: float = Field(..., description="DTW 距离")
    dtw_similarity: float = Field(..., description="DTW 相似度（0..1，越大越相似）")
    area_above: float = Field(..., description="偏差在上方（超上界）随时间积分的面积（幅度×时间）")
    area_below: float = Field(..., description="偏差在下方（低于下界）随时间积分的面积（幅度×时间）")
    is_abnormal_pack: bool = Field(..., description="是否被判定为异常包（布尔）")


class HeatmapPayload(BaseModel):
    """prepare_heatmap_json 返回结构"""
    columns: List[Union[int, str]] = Field(..., description="X 轴标签")
    matrix: List[List[Optional[float]]] = Field(
        ..., description="数值矩阵（1 x T）"
    )
    vmin: Optional[float] = Field(None, description="色标最小值")
    vmax: Optional[float] = Field(None, description="色标最大值")


class BandItem(BaseModel):
    """单个 band（lower/upper）用于前端绘图"""
    label: str = Field(..., description="例如 '1-99%' 或 '±2σ'")
    lower: List[Optional[float]] = Field(..., description="lower 数组")
    upper: List[Optional[float]] = Field(..., description="upper 数组")


class TimeseriesPayload(BaseModel):
    """prepare_pack_timeseries_json 返回结构"""
    columns: List[Union[int, str]] = Field(..., description="X 轴标签")
    current: List[Optional[float]] = Field(..., description="当前 pack 值序列")
    bands: List[BandItem] = Field(..., description="阈值带数组")
    meta: Dict[str, Any] = Field(..., description="元信息，例如 n_points, n_bands")


class PackProcessResponse(BaseModel):
    """
    最终的 endpoint 响应模型，合并 analysis + heatmap + timeseries
    """
    # 分析指标
    analysis: AnalysisResult = Field(..., description="综合分析指标")
    # heatmap 部分
    deviation_heatmap: HeatmapPayload = Field(..., description="偏差热力图 JSON")
    # timeseries 部分
    timeseries: TimeseriesPayload = Field(..., description="多带区间与 current 的时序 JSON")

class VoltSumItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time, e.g. '2025-12-27T11:30:00'")
    volt_sum: Optional[float] = Field(None, description="按时间累加的所有 pack 的 cell 电压之和")

class TempMinItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    temp_min: Optional[float] = Field(None, description="按时间取所有 pack 的电池温度最小值")

class VoltDiffItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    volt_diff: Optional[float] = Field(None, description="按时间取所有 pack 的 (max(cell_volt) - min(cell_volt))")

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

class StepNameItem(BaseModel):
    time: str = Field(..., description="ISO-format acquire_time")
    step_name: Optional[str] = Field(None, description="按时间映射的 step_name（若缺失为 null）")


class PackProcessDisplayResponse(BaseModel):
    """
    Response model for process display aggregation.

    返回按时间顺序的列表；每个列表项是 {'time': ISO_str, '<metric>': value} 形式。
    """

    volt_sum_list: List[VoltSumItem] = Field(
        default_factory=list,
        description="按时间累加的所有 pack 的 cell 电压之和列表，按时间升序"
    )
    temp_min_list: List[TempMinItem] = Field(
        default_factory=list,
        description="按时间取所有 pack 的电池温度最小值列表，按时间升序"
    )
    volt_diff_list: List[VoltDiffItem] = Field(
        default_factory=list,
        description="按时间取所有 pack 的 volt_diff 列表，按时间升序"
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

    step_name_list: List[StepNameItem] = Field(
        default_factory=list,
        description="按时间映射的 step_name 列表，按时间升序"
    )