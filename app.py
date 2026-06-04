from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION, CORS_ORIGINS
from api.routes import router as predict_router
from api.service import model_service
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega modelo no startup. Se falhar, o servidor não fica "ok" no /health.
    try:
        model_service.load()
    except Exception:
        # Mantém o serviço em estado "not ready"; /health refletirá loading.
        pass
    yield


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Checar status do servidor e do modelo",
    description=(
        "Retorna se o modelo já foi carregado no startup, qual dispositivo está sendo usado "
        "(CPU/GPU) e se o serviço está operacional para previsões."
    ),
    response_description="Status atual do modelo e do serviço.",
)
def health_check() -> HealthResponse:
    return HealthResponse(**model_service.get_status())



app.include_router(predict_router)

