"""请求/响应模型与输入校验。

约束：
* 每次请求 2 至 5000 个观测；
* 每项有唯一 ASCII ``id``、整数 ``reading`` ∈ [0, 10^9]、
  整数 ``weight`` ∈ [1, 10^6]；
* 未知字段、重复 id、越界值、类型错误一律令整次请求返回 422。

固定观测校正（``/correct/anchored``）额外要求：
* ``anchors`` 以唯一 id 指定 1 至 12 个锚点；
* 锚点 id 不得重复，且必须存在于本批观测中——违反同样返回 422。

平台段校正（``/correct/plateau``）额外要求：
* ``platform`` 以原观测 id 指定一段含至少两点的连续范围；
  id 不得重复、必须存在、且位置必须连续（顺序不限）；
* ``anchors`` 可选，规则同上（0 至 12 个）——违反同样返回 422。
"""

from __future__ import annotations

from typing import List

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

MAX_OBSERVATIONS = 5000
MIN_OBSERVATIONS = 2
MAX_READING = 10**9
MAX_WEIGHT = 10**6
MIN_ANCHORS = 1
MAX_ANCHORS = 12
MIN_PLATFORM_SIZE = 2


class Observation(BaseModel):
    """单个刀具磨损观测。拒绝一切未知字段。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    reading: int = Field(ge=0, le=MAX_READING)
    weight: int = Field(ge=1, le=MAX_WEIGHT)

    @field_validator("id")
    @classmethod
    def _ascii_only(cls, value: str) -> str:
        if not value.isascii():
            raise ValueError("id must contain ASCII characters only")
        return value


def _ensure_unique_ids(value: List[Observation]) -> List[Observation]:
    seen: set[str] = set()
    for item in value:
        if item.id in seen:
            raise ValueError(f"duplicate id: {item.id!r}")
        seen.add(item.id)
    return value


class CorrectionRequest(BaseModel):
    """整批观测；额外的顶层未知字段同样拒绝。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    observations: List[Observation] = Field(
        min_length=MIN_OBSERVATIONS, max_length=MAX_OBSERVATIONS
    )

    @field_validator("observations")
    @classmethod
    def _unique_ids(cls, value: List[Observation]) -> List[Observation]:
        return _ensure_unique_ids(value)


class AnchoredCorrectionRequest(BaseModel):
    """固定观测校正请求：观测同 /correct，另以唯一 id 指定 1～12 个锚点。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    observations: List[Observation] = Field(
        min_length=MIN_OBSERVATIONS, max_length=MAX_OBSERVATIONS
    )
    anchors: List[str] = Field(min_length=MIN_ANCHORS, max_length=MAX_ANCHORS)

    @field_validator("observations")
    @classmethod
    def _unique_ids(cls, value: List[Observation]) -> List[Observation]:
        return _ensure_unique_ids(value)

    @field_validator("anchors")
    @classmethod
    def _unique_anchor_ids(cls, value: List[str]) -> List[str]:
        seen: set[str] = set()
        for item in value:
            if item in seen:
                raise ValueError(f"duplicate anchor id: {item!r}")
            seen.add(item)
        return value

    @model_validator(mode="after")
    def _anchors_must_reference_known_ids(self) -> "AnchoredCorrectionRequest":
        known = {item.id for item in self.observations}
        for anchor in self.anchors:
            if anchor not in known:
                raise ValueError(f"unknown anchor id: {anchor!r}")
        return self


class FractionModel(BaseModel):
    """不可约分数。"""

    model_config = ConfigDict(extra="forbid")

    numerator: int
    denominator: int


class BlockModel(BaseModel):
    """连续合并块：含端点的 0 基索引与块加权均值。"""

    model_config = ConfigDict(extra="forbid")

    start_index: int
    end_index: int
    mean: FractionModel


class FittedPoint(BaseModel):
    """逐点拟合结果，保留全部原始点。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    fitted: FractionModel


class CorrectionResponse(BaseModel):
    """保序校正结果。"""

    model_config = ConfigDict(extra="forbid")

    blocks: List[BlockModel]
    fitted: List[FittedPoint]
    total_error: FractionModel


class AnchoredFittedPoint(BaseModel):
    """逐点拟合结果；``fixed`` 标出经人工复测、原值保留的固定点。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    fitted: FractionModel
    fixed: bool


class AnchoredCorrectionResponse(BaseModel):
    """固定观测校正结果：结构同 /correct，逐点标出固定点。"""

    model_config = ConfigDict(extra="forbid")

    blocks: List[BlockModel]
    fitted: List[AnchoredFittedPoint]
    total_error: FractionModel


def _ensure_unique_strs(value: List[str], what: str) -> List[str]:
    seen: set[str] = set()
    for item in value:
        if item in seen:
            raise ValueError(f"duplicate {what} id: {item!r}")
        seen.add(item)
    return value


class PlateauCorrectionRequest(BaseModel):
    """平台段校正请求：观测同 /correct，另以原观测 id 指定连续平台范围。

    ``platform`` 为至少两个观测 id（顺序不限），其位置必须构成连续段；
    ``anchors`` 可选，语义同 /correct/anchored（允许落在平台内）。
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    observations: List[Observation] = Field(
        min_length=MIN_OBSERVATIONS, max_length=MAX_OBSERVATIONS
    )
    platform: List[str] = Field(
        min_length=MIN_PLATFORM_SIZE, max_length=MAX_OBSERVATIONS
    )
    anchors: List[str] = Field(default_factory=list, max_length=MAX_ANCHORS)

    @field_validator("observations")
    @classmethod
    def _unique_ids(cls, value: List[Observation]) -> List[Observation]:
        return _ensure_unique_ids(value)

    @field_validator("platform")
    @classmethod
    def _unique_platform_ids(cls, value: List[str]) -> List[str]:
        return _ensure_unique_strs(value, "platform")

    @field_validator("anchors")
    @classmethod
    def _unique_anchor_ids(cls, value: List[str]) -> List[str]:
        return _ensure_unique_strs(value, "anchor")

    @model_validator(mode="after")
    def _ids_known_and_platform_contiguous(self) -> "PlateauCorrectionRequest":
        position_by_id = {
            item.id: index for index, item in enumerate(self.observations)
        }
        for platform_id in self.platform:
            if platform_id not in position_by_id:
                raise ValueError(f"unknown platform id: {platform_id!r}")
        for anchor in self.anchors:
            if anchor not in position_by_id:
                raise ValueError(f"unknown anchor id: {anchor!r}")
        positions = sorted(position_by_id[platform_id] for platform_id in self.platform)
        if positions[-1] - positions[0] != len(positions) - 1:
            raise ValueError("platform ids must form a contiguous range")
        return self


class PlateauFittedPoint(BaseModel):
    """逐点拟合结果；``platform`` 标出平台段内各点，``fixed`` 标出固定点。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    fitted: FractionModel
    fixed: bool
    platform: bool


class PlateauCorrectionResponse(BaseModel):
    """平台段校正结果：结构同 /correct，逐点标出平台与固定点。"""

    model_config = ConfigDict(extra="forbid")

    blocks: List[BlockModel]
    fitted: List[PlateauFittedPoint]
    total_error: FractionModel
