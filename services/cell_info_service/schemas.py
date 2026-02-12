from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, validator

class PackQuery(BaseModel):
    pack_code: List = Field(..., description="PACK编号")


class CellInfoItem(BaseModel):
    pack_code: str = Field(..., description="Pack code")
    cell_code: Optional[str] = Field(None, description="Cell code / id")
    module_in_pack: Optional[int] = Field(None, description="Module index inside pack")
    cell_in_module: Optional[int] = Field(None, description="Cell index inside module")
    capacity: Optional[float] = Field(None, description="Cell capacity (float)")
    ocv3: Optional[float] = Field(None, description="OCV3 value (float)")
    ocv4: Optional[float] = Field(None, description="OCV4 value (float)")
    acr3: Optional[float] = Field(None, description="ACR3 value (float)")
    acr4: Optional[float] = Field(None, description="ACR4 value (float)")
    k_value: Optional[float] = Field(None, description="K-value (float)")
    cell_thickness: Optional[float] = Field(None, description="Cell thickness (float)")
    weight: Optional[float] = Field(None, description="Cell weight (float)")
    cell_index: Optional[int] = Field(None, description="Global cell index in pack (from map)")


class CellInfoResponse(BaseModel):
    results: List[CellInfoItem] = Field(
        default_factory=list,
        description="List of cell records (one dict per cell) sorted by pack_code and cell_index"
    )
