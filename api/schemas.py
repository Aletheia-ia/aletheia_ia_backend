from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    device: str = Field(..., examples=["cpu"])


class PredictRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["As urnas foram fraudadas"],
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    texto: str
    label: str
    confianca: float
    prob_falso: float
    prob_verdadeiro: float
    limiar: float


class ErrorResponse(BaseModel):
    detail: str
