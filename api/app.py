from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    OPENAPI_TAGS,
)
from api.routes import router as predict_router
from api.schemas import ApiInfoResponse, HealthResponse
from api.service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
    except Exception:
        pass
    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=ApiInfoResponse,
    tags=["sistema"],
    summary="Informações da API",
    description="Retorna links para a documentação Swagger, ReDoc e o resumo das rotas.",
)
def api_info() -> ApiInfoResponse:
    return ApiInfoResponse(
        nome=API_TITLE,
        versao=API_VERSION,
        documentacao={
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        rotas=[
            {"metodo": "GET", "rota": "/", "descricao": "Informações e links da API"},
            {"metodo": "GET", "rota": "/health", "descricao": "Status do servidor e do modelo"},
            {"metodo": "POST", "rota": "/predict", "descricao": "Classificar um texto"},
            {"metodo": "GET", "rota": "/docs", "descricao": "Documentação interativa (Swagger UI)"},
            {"metodo": "GET", "rota": "/redoc", "descricao": "Documentação alternativa (ReDoc)"},
        ],
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["sistema"],
    summary="Verificar saúde do serviço",
    description=(
        "Use esta rota para saber se o modelo já foi carregado e se a API está pronta "
        "para receber requisições em `/predict`.\n\n"
        "Em deploy, consulte antes de testar o frontend — útil também para "
        "\"acordar\" o Space no Hugging Face após inatividade."
    ),
    response_description="Status atual do modelo e do dispositivo de inferência.",
    responses={
        200: {
            "description": "Serviço respondendo; verifique `model_loaded` para uso em produção.",
            "content": {
                "application/json": {
                    "examples": {
                        "pronto": {
                            "summary": "Modelo carregado",
                            "value": {
                                "status": "ok",
                                "model_loaded": True,
                                "device": "cpu",
                            },
                        },
                        "carregando": {
                            "summary": "Modelo ainda não disponível",
                            "value": {
                                "status": "loading",
                                "model_loaded": False,
                                "device": "cpu",
                            },
                        },
                    }
                }
            },
        }
    },
)
def health_check() -> HealthResponse:
    return HealthResponse(**model_service.get_status())


app.include_router(predict_router)
