from fastapi import APIRouter, HTTPException

from api.schemas import ErrorResponse, PredictRequest, PredictResponse
from api.service import model_service

router = APIRouter(tags=["classificacao"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classificar texto como FALSO ou VERDADEIRO",
    description=(
        "Recebe um texto em português e retorna a classificação de fake news eleitorais. "
        "O endpoint aplica limpeza de texto antes da inferência e usa um limiar (threshold) "
        "para decidir o rótulo final." 
    ),
    response_description="Resultado da classificação com probabilidades e limiar usado.",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Erro de validação (ex.: texto vazio ou inválido após limpeza).",
        },
        503: {
            "model": ErrorResponse,
            "description": "Modelo ainda não carregado no startup (ou falha no carregamento).",
        },
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

