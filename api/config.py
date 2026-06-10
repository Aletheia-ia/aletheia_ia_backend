import os

API_TITLE = "Aletheia IA API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API REST do classificador **Aletheia IA** — identifica se um texto em português
sobre política/eleições é provável **fake news** ou **verdadeiro**.

## Modelo

- Base: BERTimbau (`neuralmind/bert-base-portuguese-cased`)
- Saída: `FALSO` ou `VERDADEIRO` com probabilidades e confiança

## Documentação interativa

| Interface | URL |
|-----------|-----|
| Swagger UI | `/docs` |
| ReDoc | `/redoc` |
| OpenAPI JSON | `/openapi.json` |

## Rotas principais

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status do servidor e do modelo |
| `POST` | `/predict` | Classifica um texto |

## Exemplo rápido

```bash
curl -X POST https://sua-api.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{"texto": "As urnas foram fraudadas"}'
```
"""

OPENAPI_TAGS = [
    {
        "name": "sistema",
        "description": "Monitoramento e disponibilidade do serviço.",
    },
    {
        "name": "classificacao",
        "description": "Classificação de textos como FALSO ou VERDADEIRO.",
    },
]

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))

# Em produção, defina CORS_ORIGINS com o domínio real do frontend via variável de ambiente.
# Ex: CORS_ORIGINS=https://meu-frontend.com
_default_origins = "http://localhost:3000,http://localhost:5173"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", _default_origins).split(",")
