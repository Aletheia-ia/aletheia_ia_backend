from fastapi import APIRouter, HTTPException

from api.schemas import ErrorResponse, PredictRequest, PredictResponse
from api.service import model_service

router = APIRouter(tags=["classificacao"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classificar texto",
    description=(
        "Recebe um texto em português e retorna se é classificado como **FALSO** "
        "ou **VERDADEIRO**, com probabilidades e confiança.\n\n"
        "O texto passa por limpeza automática (remoção de URLs, emojis e espaços extras) "
        "antes da inferência com o BERTimbau."
    ),
    response_description="Classificação concluída com sucesso.",
    responses={
        200: {
            "description": "Texto classificado com sucesso.",
            "content": {
                "application/json": {
                    "example": {
                        "texto": "As urnas foram fraudadas",
                        "label": "FALSO",
                        "confianca": 62.9,
                        "prob_falso": 62.9,
                        "prob_verdadeiro": 37.1,
                        "limiar": 0.5,
                    }
                }
            },
        },
        422: {
            "description": "Texto vazio, inválido após limpeza ou JSON fora do schema.",
            "model": ErrorResponse,
        },
        503: {
            "description": "Modelo ainda não carregado ou serviço indisponível.",
            "model": ErrorResponse,
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
