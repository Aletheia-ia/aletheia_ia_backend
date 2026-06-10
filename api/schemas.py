from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Status operacional da API e do modelo carregado no startup."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model_loaded": True,
                    "device": "cpu",
                }
            ]
        }
    )

    status: str = Field(
        ...,
        description="Estado do serviço: `ok` quando o modelo está pronto, `loading` caso contrário.",
        examples=["ok", "loading"],
    )
    model_loaded: bool = Field(
        ...,
        description="Indica se o modelo BERTimbau já foi carregado na memória.",
    )
    device: str = Field(
        ...,
        description="Dispositivo usado na inferência: `cpu` ou `cuda`.",
        examples=["cpu", "cuda"],
    )


class PredictRequest(BaseModel):
    """Texto a ser classificado e limiar opcional para VERDADEIRO."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texto": "As urnas foram fraudadas",
                    "threshold": 0.5,
                },
                {
                    "texto": "Pesquisa Datafolha mostra empate técnico entre candidatos",
                    "threshold": 0.5,
                },
            ]
        }
    )

    texto: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Afirmação ou trecho de notícia em português a ser analisado.",
        examples=["As urnas foram fraudadas"],
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Limiar de probabilidade para classificar como VERDADEIRO. "
            "Valores mais altos exigem mais confiança para marcar como verdadeiro."
        ),
        examples=[0.5],
    )


class PredictResponse(BaseModel):
    """Resultado da classificação com probabilidades por classe."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texto": "As urnas foram fraudadas",
                    "label": "FALSO",
                    "confianca": 62.9,
                    "prob_falso": 62.9,
                    "prob_verdadeiro": 37.1,
                    "limiar": 0.5,
                }
            ]
        }
    )

    texto: str = Field(..., description="Texto enviado após limpeza (URLs e emojis removidos).")
    label: str = Field(
        ...,
        description="Classificação final: `FALSO` ou `VERDADEIRO`.",
        examples=["FALSO", "VERDADEIRO"],
    )
    confianca: float = Field(
        ...,
        description="Confiança da classe escolhida, em percentual (0–100).",
        examples=[62.9],
    )
    prob_falso: float = Field(..., description="Probabilidade da classe FALSO (%).")
    prob_verdadeiro: float = Field(..., description="Probabilidade da classe VERDADEIRO (%).")
    limiar: float = Field(..., description="Limiar aplicado na classificação.")


class ErrorResponse(BaseModel):
    """Resposta padrão de erro da API."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"detail": "Texto inválido ou vazio após limpeza."},
                {"detail": "Modelo ainda não foi carregado."},
            ]
        }
    )

    detail: str = Field(..., description="Mensagem descritiva do erro.")


class ApiInfoResponse(BaseModel):
    """Links úteis para explorar a API."""

    nome: str = Field(..., description="Nome da API.")
    versao: str = Field(..., description="Versão publicada.")
    documentacao: dict[str, str] = Field(
        ...,
        description="URLs da documentação interativa e do schema OpenAPI.",
    )
    rotas: list[dict[str, str]] = Field(..., description="Resumo das rotas disponíveis.")
