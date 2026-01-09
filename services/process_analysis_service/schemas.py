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

class StepRangeItem(BaseModel):
    step_id: Optional[Any] = Field(
        None, description="步骤 ID（可能为 int/float/str），如果未知则为 null"
    )
    step_name: Optional[str] = Field(
        None, description="步骤名称，如 '恒流充电'，如果未知则为 null"
    )
    range: List[Optional[str]] = Field(
        ..., description="开始和结束时间的 ISO 字符串或可被序列化为字符串的时间，格式为 [start_iso, end_iso]"
    )

class PackProcessDisplayResponse(BaseModel):

    voltage_series: List[Optional[float]] = Field(
        default_factory=list,
        description="每行所有 cell 电压之和（若该行全部缺失则为 null）",
    )
    temperature_series: List[Optional[float]] = Field(
        default_factory=list,
        description="每行所有电池温度的最小值（若该行全部缺失则为 null）",
    )

    charge_energy_list: List[Optional[float]] = Field(
        default_factory=list, description="每行 charge_energy（若缺失为 null）"
    )
    discharge_energy_list: List[Optional[float]] = Field(
        default_factory=list, description="每行 discharge_energy（若缺失为 null）"
    )
    charge_capacity_list: List[Optional[float]] = Field(
        default_factory=list, description="每行 charge_capacity（若缺失为 null）"
    )
    discharge_capacity_list: List[Optional[float]] = Field(
        default_factory=list, description="每行 discharge_capacity（若缺失为 null）"
    )

    volt_diff_list: List[Optional[float]] = Field(
        default_factory=list,
        description="每行所有 cell 电压的 (max - min)，若全部缺失则为 null",
    )

    time_list: List[Optional[str]] = Field(
        default_factory=list,
        description="每行的时间戳（ISO 格式字符串），若缺失则为 null",
    )

    all_segments: List[StepRangeItem] = Field(
        default_factory=list,
        description="按 (step_id, step_name) 分组后返回的步骤区间列表，每项包含 step_id, step_name 和 [start, end] 时间范围",
    )