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

**2. Fake.br-Corpus** via [roneysco/Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus), corpus acadêmico publicado na conferência PROPOR 2018, composto por notícias verdadeiras e falsas em português brasileiro. Essa fonte originou os registros classificados como `True`.

### Palavras-chave utilizadas

As buscas foram realizadas com as seguintes palavras-chave de tema eleitoral:

`eleição` · `Bolsonaro` · `lula` · `pt` · `campanha` · `urna` · `voto` · `fraude`

### Pipeline de coleta

```
Fact Check Explorer
704 registros coletados
 ↓ remoção de duplicatas
~600 registros únicos
 ↓ filtro de idioma (somente português)
421 registros em português
 ↓ descarte de vereditos indefinidos
632 registros Fake

+

Fake.br-Corpus
3600 notícias verdadeiras disponíveis
 ↓ filtro por palavras-chave eleitorais
632 registros True selecionados

=

1264 registros no dataset final
```

### Estrutura do dataset

| Coluna | Descrição |
|--------|-----------|
| `texto` | A afirmação verificada ou notícia coletada |
| `label` | Rotulo final: `Fake` ou `True` |
| `verdict_original` | Veredito original da agência (ex: "Falso", "Enganoso") |
| `fonte` | Nome da agência ou corpus de origem |
| `url_original` | Link para a checagem completa |
| `data_checagem` | Data em que a checagem foi publicada |
| `tags` | Tags temáticas associadas |
| `palavra_chave` | Keyword que originou o registro |

### Criterios de classificação dos rotulos

**Fake** - vereditos que contêm: `falso`, `enganoso`, `distorcido`, `misleading`, `false`, `incorreto`, entre outros.

**True** - notícias verdadeiras extraídas do Fake.br-Corpus, corpus validado academicamente.

Registros com vereditos ambíguos ou sem classificação clara foram descartados para garantir a qualidade do treino.

### Distribuição do dataset

| Label | Quantidade | Percentual |
|-------|------------|------------|
| Fake  | 632        | 50%        |
| True  | 632        | 50%        |
| Total | 1264       | 100%       |

### Como reproduzir a coleta

```bash
# Instalar dependências
pip install git+https://github.com/GONZOsint/factcheckexplorer.git
pip install pandas langdetect requests

# Coletar os Fake via Fact Check Explorer
python dataset/consultar_fact_check.py

# Coletar e adicionar os True via Fake.br-Corpus
python dataset/consultar_true.py
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