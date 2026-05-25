# Guia de Implementação — API FastAPI (Aletheia IA)

Este documento divide a criação da API REST do classificador de fake news em **3 passos de complexidade equivalente**. Cada passo envolve **2 arquivos**, cerca de **60–80 linhas de código** e uma **validação própria** antes de avançar.

**Stack:** Python + [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn

---

## Visão geral dos 3 passos

| Passo | Foco | Arquivos | Entrega |
|-------|------|----------|---------|
| **1** | Configuração e contratos HTTP | `config.py`, `schemas.py` | Tipos e variáveis centralizados |
| **2** | Serviço do modelo e ciclo de vida | `service.py`, `main.py` (parcial) | Modelo carregado no startup |
| **3** | Rota de predição e tratamento de erros | `routes.py`, `main.py` (final) | API completa para o frontend |

```
projeto_ml/
├── api/
│   ├── __init__.py
│   ├── config.py        ← Passo 1
│   ├── schemas.py       ← Passo 1
│   ├── service.py       ← Passo 2
│   ├── routes.py        ← Passo 3
│   └── main.py          ← Passo 2 (parcial) + Passo 3 (final)
├── model/
├── predict.py
└── requirements.txt
```

---

## Pré-requisitos

Confirme que o modelo foi treinado antes de iniciar:

```bash
cd projeto_ml
python train.py --data dataset/treino.csv
```

A pasta `model/` deve conter os artefatos do Hugging Face (`config.json`, pesos do modelo, tokenizer).

Instale as dependências da API **uma vez**, antes do Passo 1:

```bash
pip install fastapi "uvicorn[standard]"
```

Adicione ao `requirements.txt`:

```
fastapi
uvicorn[standard]
```

Crie a pasta base:

```bash
mkdir -p projeto_ml/api
touch projeto_ml/api/__init__.py
```

---

## Passo 1 — Configuração e contratos HTTP

**Objetivo:** centralizar variáveis de ambiente e definir os schemas Pydantic que serão usados em todas as rotas.

**Arquivos deste passo:** `api/config.py`, `api/schemas.py`  
**Complexidade:** ~70 linhas · 2 arquivos novos · nenhuma rota ainda

### 1.1 Criar `api/config.py`

Responsável por ler variáveis de ambiente com valores padrão. Evita espalhar `os.getenv` pelo código.

```python
import os

API_TITLE = "Aletheia IA API"
API_DESCRIPTION = "Classificador de fake news eleitorais em português"
API_VERSION = "1.0.0"

MODEL_DIR = os.getenv("MODEL_DIR", "model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))

# Em produção, substitua "*" pelo domínio do frontend
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
```

### 1.2 Criar `api/schemas.py`

Define o formato de entrada e saída de **todas** as rotas da API.

```python
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
```

### 1.3 Validar (sem subir servidor)

Confirme que os módulos importam corretamente:

```bash
cd projeto_ml
python -c "from api.config import MODEL_DIR, DEFAULT_THRESHOLD; from api.schemas import PredictRequest; print(MODEL_DIR, PredictRequest(texto='teste').threshold)"
```

Saída esperada:

```
model 0.5
```

**Critério de conclusão do Passo 1:** `config.py` e `schemas.py` existem, importam sem erro e cobrem health, predict e erro.

---

## Passo 2 — Serviço do modelo e ciclo de vida

**Objetivo:** carregar o BERTimbau uma única vez na inicialização e expor o endpoint `GET /health` com status real do modelo.

**Arquivos deste passo:** `api/service.py`, `api/main.py` (versão parcial)  
**Complexidade:** ~75 linhas · 1 arquivo novo + 1 parcial · 1 rota funcional

### 2.1 Criar `api/service.py`

Encapsula o carregamento do modelo reutilizando funções de `predict.py`. Neste passo, implemente apenas **load** e **status** — a predição fica para o Passo 3.

```python
import os

import torch
from transformers import AutoTokenizer

from api.config import MODEL_DIR
from predict import load_model

class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(
                f"Modelo não encontrado em '{MODEL_DIR}'. "
                "Execute: python train.py --data dataset/treino.csv"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
        self.model = load_model(MODEL_DIR, torch_dtype=None)
        self.model.to(self.device)
        self.model.eval()

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_status(self) -> dict:
        return {
            "status": "ok" if self.is_ready() else "loading",
            "model_loaded": self.is_ready(),
            "device": self.device.type,
        }


model_service = ModelService()
```

### 2.2 Criar `api/main.py` (versão parcial)

Monta a aplicação FastAPI, CORS, lifespan e a rota de health. Ainda **sem** `POST /predict`.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION, CORS_ORIGINS
from api.schemas import HealthResponse
from api.service import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
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


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(**model_service.get_status())
```

### 2.3 Subir e validar

```bash
cd projeto_ml
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Em outro terminal:

```bash
curl http://localhost:8000/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

Abra também http://localhost:8000/docs — deve listar apenas `GET /health`.

**Critério de conclusão do Passo 2:** servidor sobe, modelo carrega no startup e `/health` retorna `model_loaded: true` com o device correto.

---

## Passo 3 — Rota de predição e tratamento de erros

**Objetivo:** expor `POST /predict` com validação, mapeamento de exceções e resposta tipada para o frontend.

**Arquivos deste passo:** `api/routes.py`, atualização de `api/service.py` e `api/main.py`  
**Complexidade:** ~75 linhas · 1 arquivo novo + 2 atualizações · API completa

### 3.1 Adicionar `predict()` em `api/service.py`

Acrescente ao final da classe `ModelService` (imports extras no topo):

```python
from dataclasses import dataclass

from api.config import DEFAULT_THRESHOLD, MAX_LENGTH
from predict import clean_text, run_inference


@dataclass
class PredictionResult:
    texto: str
    label: str
    confianca: float
    prob_falso: float
    prob_verdadeiro: float
    limiar: float
```

Método a adicionar na classe:

```python
    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> PredictionResult:
        if not self.is_ready():
            raise RuntimeError("Modelo ainda não foi carregado.")

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Texto inválido ou vazio após limpeza.")

        label, confidence, prob_fake, prob_true = run_inference(
            self.model,
            self.tokenizer,
            cleaned,
            self.device,
            MAX_LENGTH,
            threshold,
        )

        return PredictionResult(
            texto=cleaned,
            label=label,
            confianca=round(confidence, 1),
            prob_falso=round(prob_fake, 1),
            prob_verdadeiro=round(prob_true, 1),
            limiar=threshold,
        )
```

### 3.2 Criar `api/routes.py`

Separa a lógica HTTP da predição do arquivo principal, mantendo `main.py` enxuto.

```python
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
```

### 3.3 Atualizar `api/main.py`

Registre o router criado no passo anterior:

```python
from api.routes import router as predict_router

# ... código existente do Passo 2 ...

app.include_router(predict_router)
```

### 3.4 Validar

Reinicie o servidor e teste a classificação:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "As urnas foram fraudadas"}'
```

Resposta esperada:

```json
{
  "texto": "As urnas foram fraudadas",
  "label": "FALSO",
  "confianca": 93.2,
  "prob_falso": 93.2,
  "prob_verdadeiro": 6.8,
  "limiar": 0.5
}
```

Teste de erro (texto vazio → `422`):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "   "}'
```

Teste no frontend:

```javascript
const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ texto: "As urnas foram fraudadas" }),
});

const data = await response.json();
console.log(data.label, data.confianca);
```

**Critério de conclusão do Passo 3:** `POST /predict` classifica textos, erros retornam status adequado e `/docs` lista as duas rotas.

---

## Resumo por passo

| Passo | Arquivos | Linhas (~) | Rota(s) | Validação |
|-------|----------|------------|---------|-----------|
| 1 | `config.py`, `schemas.py` | 70 | — | Import Python |
| 2 | `service.py`, `main.py` | 75 | `GET /health` | curl `/health` |
| 3 | `routes.py` + updates | 75 | `POST /predict` | curl `/predict` |

---

## Mapa final de endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status do servidor, modelo e device |
| `POST` | `/predict` | Classifica um texto |
| `GET` | `/docs` | Documentação Swagger |
| `GET` | `/redoc` | Documentação ReDoc |

### Códigos de erro em `/predict`

| Status | Quando ocorre |
|--------|---------------|
| `422` | Texto vazio, inválido após limpeza ou JSON fora do schema |
| `503` | Modelo não carregado |

---

## Execução

```bash
cd projeto_ml
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Variáveis de ambiente (definidas em `config.py`):

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MODEL_DIR` | `model` | Caminho do modelo treinado |
| `MAX_LENGTH` | `128` | Limite de tokens |
| `THRESHOLD` | `0.5` | Limiar padrão para VERDADEIRO |
| `CORS_ORIGINS` | `*` | Origens permitidas (separadas por vírgula) |
| `API_HOST` | `0.0.0.0` | Host do Uvicorn |
| `API_PORT` | `8000` | Porta do Uvicorn |

---

## Checklist final

- [ ] **Passo 1:** `config.py` e `schemas.py` importam sem erro
- [ ] **Passo 2:** `/health` retorna `model_loaded: true`
- [ ] **Passo 3:** `/predict` classifica e trata erros corretamente
- [ ] `requirements.txt` inclui `fastapi` e `uvicorn`
- [ ] Frontend consegue chamar a API sem erro de CORS

---

## Próximos passos (opcionais)

- Endpoint de batch para múltiplos textos
- Autenticação (API key ou JWT)
- Docker para deploy
- Endpoint de feedback baseado em `interactive_feedback.py`

Para detalhes do modelo e pipeline de ML, consulte [DOCUMENTACAO.md](./DOCUMENTACAO.md).
