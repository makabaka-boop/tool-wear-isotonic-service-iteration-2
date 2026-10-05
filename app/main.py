"""FastAPI 入口：接收普通 JSON，返回保序校正结果。"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .isotonic import (
    FitResult,
    InfeasibleAnchorsError,
    fit_isotonic,
    fit_isotonic_anchored,
)
from .schemas import (
    AnchoredCorrectionRequest,
    AnchoredCorrectionResponse,
    AnchoredFittedPoint,
    BlockModel,
    CorrectionRequest,
    CorrectionResponse,
    FittedPoint,
    FractionModel,
)

app = FastAPI(
    title="Tool Wear Isotonic Correction Service",
    version="1.0.0",
)


def _block_models(result: FitResult) -> list[BlockModel]:
    return [
        BlockModel(
            start_index=block.start,
            end_index=block.end,
            mean=FractionModel(
                numerator=block.mean_num,
                denominator=block.mean_den,
            ),
        )
        for block in result.blocks
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/correct", response_model=CorrectionResponse)
def correct(request: CorrectionRequest) -> CorrectionResponse:
    observations = request.observations
    readings = [o.reading for o in observations]
    weights = [o.weight for o in observations]

    result = fit_isotonic(readings, weights)

    fitted = [
        FittedPoint(
            id=observations[i].id,
            fitted=FractionModel(
                numerator=result.fitted_num[i],
                denominator=result.fitted_den[i],
            ),
        )
        for i in range(len(observations))
    ]
    return CorrectionResponse(
        blocks=_block_models(result),
        fitted=fitted,
        total_error=FractionModel(
            numerator=result.total_error_num,
            denominator=result.total_error_den,
        ),
    )


@app.post("/correct/anchored", response_model=AnchoredCorrectionResponse)
def correct_anchored(
    request: AnchoredCorrectionRequest,
) -> AnchoredCorrectionResponse:
    observations = request.observations
    readings = [o.reading for o in observations]
    weights = [o.weight for o in observations]
    position_by_id = {o.id: i for i, o in enumerate(observations)}
    anchor_positions = [position_by_id[anchor] for anchor in request.anchors]

    try:
        result = fit_isotonic_anchored(readings, weights, anchor_positions)
    except InfeasibleAnchorsError as exc:
        # 锚点读数按观测顺序下降：等式约束不可满足，不发布部分拟合
        raise HTTPException(
            status_code=409,
            detail={
                "code": "INFEASIBLE_ANCHORS",
                "conflict": [
                    observations[exc.left].id,
                    observations[exc.right].id,
                ],
            },
        ) from exc

    fixed_positions = set(anchor_positions)
    fitted = [
        AnchoredFittedPoint(
            id=observations[i].id,
            fitted=FractionModel(
                numerator=result.fitted_num[i],
                denominator=result.fitted_den[i],
            ),
            fixed=i in fixed_positions,
        )
        for i in range(len(observations))
    ]
    return AnchoredCorrectionResponse(
        blocks=_block_models(result),
        fitted=fitted,
        total_error=FractionModel(
            numerator=result.total_error_num,
            denominator=result.total_error_den,
        ),
    )
