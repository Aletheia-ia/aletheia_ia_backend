from fastapi import APIRouter, HTTPException

from api.schemas import ErrorResponse, PredictRequest, PredictResponse
from api.service import model_service

router = APIRouter(tags=["classificacao"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = model_service.predict(payload.texto, payload.threshold)
        return PredictResponse(**result.__dict__)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

