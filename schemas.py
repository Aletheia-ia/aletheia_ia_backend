from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"], description="Status geral do serviço.")
    model_loaded: bool = Field(
        ..., examples=[True, False], description="Indica se o modelo está carregado na memória."
    )
    device: str = Field(..., examples=["cpu"], description="Dispositivo usado na inferência (cpu/cuda).")



class PredictRequest(BaseModel):
    texto: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["As urnas foram fraudadas"],
        description="A afirmação/notícia a ser classificada.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        examples=[0.5, 0.6],
        description=(
            "Limiar para classificar como VERDADEIRO. Probabilidades acima desse valor "
            "tendem a gerar rótulo VERDADEIRO; caso contrário, FALSO."
        ),
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "texto": "As urnas foram fraudadas",
                    "threshold": 0.5,
                },
                {
                    "texto": "O Brasil realizou eleicoes em 2022",
                    "threshold": 0.6,
                },
            ]
        }



class PredictResponse(BaseModel):
    texto: str = Field(description="Texto após limpeza/padronização interna.")
    label: str = Field(description="Rótulo final: FALSO ou VERDADEIRO.")
    confianca: float = Field(description="Confiança (probabilidade) associada ao rótulo final.")
    prob_falso: float = Field(description="Probabilidade estimada para FALSO.")
    prob_verdadeiro: float = Field(description="Probabilidade estimada para VERDADEIRO.")
    limiar: float = Field(description="Valor de threshold utilizado na inferência.")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "texto": "As urnas foram fraudadas",
                    "label": "FALSO",
                    "confianca": 93.2,
                    "prob_falso": 93.2,
                    "prob_verdadeiro": 6.8,
                    "limiar": 0.5,
                },
                {
                    "texto": "O Brasil realizou eleicoes em 2022",
                    "label": "VERDADEIRO",
                    "confianca": 88.1,
                    "prob_falso": 11.9,
                    "prob_verdadeiro": 88.1,
                    "limiar": 0.6,
                },
            ]
        }



class ErrorResponse(BaseModel):
    detail: str = Field(description="Mensagem detalhando o erro retornado pela API.")

