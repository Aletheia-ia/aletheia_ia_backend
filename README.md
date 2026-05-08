# Aletheia IA - Backend (Classificador de Fake News)

Repositório do classificador de fake news com aprendizado supervisionado, desenvolvido para as disciplinas de Machine Learning e Inteligência Artificial.
O frontend do projeto está disponível em [aletheia_ia_frontend](https://github.com/Aletheia-ia/aletheia_ia_frontend).

---

## Integrantes do Grupo

| Nome |
|------|
| Arthur de Aquino |
| Denilson Santos |
| Luan Vinicius |
| Gabriel Torres |
| Henrique Estrela |
| Matheus Kaick |
| Rafael Pinto |
| Valnei Sousa |

---

## Como o Dataset Foi Preparado

### Fontes dos dados

O dataset foi construído a partir de duas fontes complementares:

**1. Google Fact Check Explorer** via [GONZOsint/factcheckexplorer](https://github.com/GONZOsint/factcheckexplorer), que agrega checagens de fatos realizadas por agências como Agência Lupa, AosFatos e G1 Fato ou Fake. Essa fonte originou os registros classificados como `Fake`.

**2. NewsAPI** via [newsapi.org](https://newsapi.org), que agrega notícias publicadas por portais jornalísticos confiáveis em português. Essa fonte originou os registros classificados como `True`.

### Palavras-chave utilizadas

As buscas foram realizadas com palavras-chave de tema eleitoral e político:

`eleição` · `Bolsonaro` · `lula` · `pt` · `campanha` · `fraude` · `STF` · `Sergio Moro` · `pesquisa eleitoral` · `candidato` · `reeleição` · `corrupção` · `Dilma` · `Michelle Bolsonaro` · `Eduardo Bolsonaro` · `Flavio Bolsonaro` · `Ciro Gomes` · `Haddad` · `Temer` · `Gleisi` · `Datafolha` · `golpe` · `censura` · `preso político` · `indiciado` · `delação` · `mensalão` · `propina` · `ministro`

### Pipeline de coleta

```
Fact Check Explorer
coleta por palavra-chave
 ↓ remoção de duplicatas
 ↓ filtro de idioma (somente português)
 ↓ descarte de vereditos indefinidos
 ↓ filtro de relevância política
508 registros Fake

+

NewsAPI
coleta por palavra-chave
 ↓ filtro de relevância política
 ↓ equilibrio por keyword com os Fake
508 registros True

=

1016 registros no dataset final
```

### Estrutura do dataset

| Coluna | Descrição |
|--------|-----------|
| `texto` | A afirmação verificada ou notícia coletada |
| `label` | Rotulo final: `Fake` ou `True` |
| `verdict_original` | Veredito original da agência ou indicação da fonte |
| `fonte` | Nome da agência ou portal de origem |
| `url_original` | Link para a checagem ou notícia completa |
| `data_checagem` | Data de publicação |
| `tags` | Tags temáticas associadas |
| `palavra_chave` | Keyword que originou o registro |

### Criterios de classificação dos rotulos

**Fake** - vereditos das agências de fact-checking que contêm: `falso`, `enganoso`, `distorcido`, `misleading`, `false`, `incorreto`, entre outros. Registros com vereditos ambíguos foram descartados.

**True** - notícias publicadas por portais jornalísticos confiáveis coletadas via NewsAPI, filtradas por relevância política.

### Distribuição do dataset

| Label | Quantidade | Percentual |
|-------|------------|------------|
| Fake  | 508        | 50%        |
| True  | 508        | 50%        |
| Total | 1016       | 100%       |

### Como reproduzir a coleta

```bash
# Instalar dependências
pip install git+https://github.com/GONZOsint/factcheckexplorer.git
pip install pandas langdetect requests python-dotenv

# Configurar a API key da NewsAPI
# Crie um arquivo .env na pasta dataset/ com o conteúdo:
# NEWSAPI_KEY=sua_chave_aqui

# Rodar o pipeline completo
python dataset/gerar_dataset.py
```

---

## Algoritmo Utilizado

> A ser preenchido.

---

## Como Foi Realizado o Treinamento

> A ser preenchido.

---

## Metricas de Avaliacao

> A ser preenchido.