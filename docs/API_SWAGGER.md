# Documentação da API — Swagger / OpenAPI

A API Aletheia IA expõe documentação interativa automaticamente via **FastAPI**.

---

## Onde visualizar as rotas

### Local (desenvolvimento)

Com o servidor rodando (`python start_api.py`):

| Interface | URL |
|-----------|-----|
| **Swagger UI** (recomendado) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| OpenAPI JSON | http://localhost:8000/openapi.json |
| Índice da API | http://localhost:8000/ |

### Deploy no Hugging Face Spaces

> **Importante:** se `/docs` retornar **404**, o Space não está rodando.
> Corrija o deploy antes (veja `docs/DEPLOY_HUGGINGFACE.md`).

Substitua pela URL do seu Space:

| Interface | URL |
|-----------|-----|
| **Swagger UI** | `https://sairth-aletheia-api.hf.space/docs` |
| ReDoc | `https://seu-usuario-aletheia-api.hf.space/redoc` |
| OpenAPI JSON | `https://seu-usuario-aletheia-api.hf.space/openapi.json` |
| Índice da API | `https://seu-usuario-aletheia-api.hf.space/` |

---

## Rotas disponíveis

### `GET /`

Retorna nome, versão, links da documentação e lista resumida de rotas.

### `GET /health` — Sistema

Verifica se o modelo BERTimbau foi carregado.

**Resposta 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

| Campo | Descrição |
|-------|-----------|
| `status` | `ok` = pronto · `loading` = modelo não carregado |
| `model_loaded` | `true` quando pode usar `/predict` |
| `device` | `cpu` ou `cuda` |

---

### `POST /predict` — Classificação

Classifica um texto como `FALSO` ou `VERDADEIRO`.

**Corpo da requisição:**
```json
{
  "texto": "As urnas foram fraudadas",
  "threshold": 0.5
}
```

| Campo | Tipo | Obrigatório | Descrição |
|-------|------|-------------|-----------|
| `texto` | string | Sim | Texto em português (1–2000 caracteres) |
| `threshold` | float | Não | Limiar para VERDADEIRO (0.0–1.0, padrão 0.5) |

**Resposta 200:**
```json
{
  "texto": "As urnas foram fraudadas",
  "label": "FALSO",
  "confianca": 62.9,
  "prob_falso": 62.9,
  "prob_verdadeiro": 37.1,
  "limiar": 0.5
}
```

**Erros:**

| Status | Quando ocorre |
|--------|---------------|
| `422` | Texto vazio ou inválido após limpeza |
| `503` | Modelo ainda não carregado |

---

## Como testar pelo Swagger UI

1. Abra `/docs` (local ou no deploy).
2. Expanda a rota desejada.
3. Clique em **Try it out**.
4. Preencha o JSON de exemplo.
5. Clique em **Execute**.
6. Veja o código de status e o corpo da resposta abaixo.

Para `/predict`, use exemplos como:

- `"As urnas foram fraudadas"` → tende a `FALSO`
- `"Pesquisa Datafolha mostra empate técnico entre candidatos"` → tende a `VERDADEIRO`

---

## Integração com o frontend

```javascript
const API_URL = "https://seu-usuario-aletheia-api.hf.space";

const response = await fetch(`${API_URL}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    texto: "Texto para verificar",
    threshold: 0.5,
  }),
});

const resultado = await response.json();
console.log(resultado.label, resultado.confianca);
```

---

## Exportar a especificação OpenAPI

Baixe o schema para importar no Postman, Insomnia ou outras ferramentas:

```bash
curl -o openapi.json https://sua-api.com/openapi.json
```

No Postman: **Import → File → openapi.json**.
