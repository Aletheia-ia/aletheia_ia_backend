# Deploy no Hugging Face Spaces (gratuito)

Guia para publicar a API Aletheia IA no [Hugging Face Spaces](https://huggingface.co/spaces) com Docker.

---

## Pré-requisitos

1. Conta gratuita em [huggingface.co/join](https://huggingface.co/join)
2. Modelo treinado localmente (`python train.py --data dataset/treino.csv`)
3. Pasta `model/` gerada após o treino
4. Repositório no GitHub (este backend)

---

## Passo 1 — Publicar o modelo no Hub

O modelo **não vai no Git** (está no `.gitignore`). Publique-o como repositório de modelo no Hugging Face:

```bash
pip install huggingface_hub

# Login (crie um token em https://huggingface.co/settings/tokens)
hf auth login

# Crie o repositório e envie os arquivos
hf upload seu-usuario/aletheia-bert ./model --repo-type model
```

Substitua `seu-usuario` pelo seu usuário do Hugging Face.

Confirme em: `https://huggingface.co/seu-usuario/aletheia-bert`

---

## Passo 2 — Preparar o README do Space

No **topo** do `README.md` do repositório, adicione este bloco YAML (antes de qualquer outro conteúdo):

```yaml
---
title: Aletheia IA API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

O Hugging Face usa esse bloco para saber que o Space roda com Docker na porta 7860.

---

## Passo 3 — Criar o Space

1. Acesse [huggingface.co/new-space](https://huggingface.co/new-space)
2. Preencha:
   - **Space name:** `aletheia-api` (ou outro nome)
   - **License:** escolha uma (ex.: MIT)
   - **SDK:** Docker
   - **Hardware:** CPU basic (gratuito — 16 GB RAM)
   - **Visibility:** Public
3. Em **Create Space**, conecte ao repositório GitHub `aletheia_ia_backend`
4. Aguarde o build (5–15 min na primeira vez)

URL final: `https://huggingface.co/spaces/seu-usuario/aletheia-api`

A API fica em: `https://seu-usuario-aletheia-api.hf.space`

---

## Passo 4 — Variáveis de ambiente no Space

Em **Settings → Variables and secrets → Variables**, adicione:

| Variável | Valor | Exemplo |
|----------|-------|---------|
| `HF_MODEL_ID` | Repositório do modelo no Hub | `seu-usuario/aletheia-bert` |
| `CORS_ORIGINS` | URL do frontend | `https://seu-app.vercel.app,http://localhost:5173` |

Opcionais: `THRESHOLD`, `MAX_LENGTH`.

Salve e faça **Factory rebuild** em Settings se o Space já estiver rodando.

---

## Passo 5 — Testar a API

```bash
# Health check
curl https://seu-usuario-aletheia-api.hf.space/health

# Predição
curl -X POST https://seu-usuario-aletheia-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "As urnas foram fraudadas"}'
```

Documentação interativa: `https://seu-usuario-aletheia-api.hf.space/docs`

---

## Passo 6 — Configurar o frontend

No projeto [aletheia_ia_frontend](https://github.com/Aletheia-ia/aletheia_ia_frontend), aponte a URL da API:

```env
VITE_API_URL=https://seu-usuario-aletheia-api.hf.space
```

Exemplo de consumo:

```javascript
const API_URL = import.meta.env.VITE_API_URL;

const response = await fetch(`${API_URL}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ texto: "Texto para verificar" }),
});
```

---

## Arquivos de deploy neste repositório

| Arquivo | Função |
|---------|--------|
| `Dockerfile` | Imagem Docker para o Space |
| `requirements-spaces.txt` | Dependências leves (CPU only) |
| `api/service.py` | Carrega modelo local ou do Hub via `HF_MODEL_ID` |

---

## Limitações do tier gratuito

- O Space **dorme após ~48h sem uso** — a primeira requisição pode demorar 30s–2min
- Repositório do Space precisa ser **público** no plano gratuito
- Disco **não persiste** entre reinícios (por isso o modelo fica no Hub, não na imagem)

**Dica:** antes de uma apresentação, abra `/health` no navegador para “acordar” o Space.

---

## Problemas comuns

| Problema | Solução |
|----------|---------|
| **404 em `/docs`, `/health` ou `/`** | Veja seção abaixo — o container provavelmente não subiu |
| `model_loaded: false` | Confira `HF_MODEL_ID` e se o modelo está público no Hub |
| Erro de CORS no frontend | Adicione a URL exata do frontend em `CORS_ORIGINS` |
| Build falha por memória | Use `requirements-spaces.txt` (não o `requirements.txt` completo) |
| Space não sobe | Verifique o bloco YAML no topo do `README.md` com `sdk: docker` |

---

## Erro 404 no deploy (página do Hugging Face)

Se aparecer **"404 — Sorry, we can't find the page"** com o emoji do Hugging Face,
a API **não está rodando** no Space. Não é problema do Swagger em si.

### Checklist de correção

1. **Confirme o Space** em `https://huggingface.co/spaces/Sairth/aletheia-api`
2. Em **Settings → General**:
   - **SDK:** `Docker`
   - **App port:** `7860`
3. Em **Settings → Variables**, adicione:
   - `HF_MODEL_ID` = `Sairth/aletheia-bert`
4. Faça **push** do `README.md` com o bloco YAML (`sdk: docker`, `app_port: 7860`)
5. Em **Settings**, clique em **Factory rebuild**
6. Abra a aba **Logs** e aguarde aparecer:
   ```
   Application startup complete
   Uvicorn running on http://0.0.0.0:7860
   ```

### URLs corretas após o Space subir

| Recurso | URL |
|---------|-----|
| Swagger | `https://sairth-aletheia-api.hf.space/docs` |
| Health | `https://sairth-aletheia-api.hf.space/health` |
| Raiz | `https://sairth-aletheia-api.hf.space/` |

### Se o Space ainda não existir

Crie pelo terminal:

```bash
hf repos create Sairth/aletheia-api --type space --space-sdk docker --flavor cpu-basic \
  -e HF_MODEL_ID=Sairth/aletheia-bert

# Envie os arquivos do projeto para o Space
hf upload Sairth/aletheia-api . . --repo-type space \
  --include "Dockerfile" --include "requirements-spaces.txt" --include "api/*" --include "lib/*" --include "README.md"
```

Ou conecte o repositório GitHub em **Settings → Repository**.
