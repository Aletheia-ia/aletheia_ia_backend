# Aletheia IA - Backend (Classificador de Fake News)

Repositório do classificador de fake news com aprendizado supervisionado, desenvolvido para a disciplina de Machine Learning.  
O frontend do projeto está disponível em [aletheia_ia_frontend](https://github.com/Aletheia-ia/aletheia_ia_frontend).

---

## 👥 Integrantes do Grupo

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

### Fonte dos dados
Os dados foram coletados via [Google Fact Check Explorer](https://github.com/GONZOsint/factcheckexplorer), uma ferramenta que agrega checagens de fatos realizadas por agências de fact-checking reconhecidas como **Agência Lupa**, **AosFatos** e **G1 Fato ou Fake**.

### Palavras-chave utilizadas
As buscas foram realizadas com as seguintes palavras-chave de tema eleitoral:

`eleição` · `Bolsonaro` · `lula` · `pt` · `campanha` · `urna` · `voto` · `fraude`

### Pipeline de coleta

```
704 registros coletados
 ↓ remoção de duplicatas
~600 registros únicos
 ↓ filtro de idioma (somente português)
421 registros em português
 ↓ descarte de vereditos indefinidos
382 registros no dataset final
```

### Estrutura do dataset

| Coluna | Descrição |
|--------|-----------|
| `texto` | A afirmação verificada pela agência |
| `label` | Rótulo final: `Fake` ou `True` |
| `verdict_original` | Veredito original da agência (ex: "Falso", "Enganoso") |
| `fonte` | Nome da agência de fact-checking |
| `url_original` | Link para a checagem completa |
| `data_checagem` | Data em que a checagem foi publicada |
| `tags` | Tags temáticas associadas |
| `palavra_chave` | Keyword que originou o registro |

### Critérios de classificação dos rótulos

Os vereditos originais das agências foram padronizados da seguinte forma:

**Fake** - vereditos que contêm: `falso`, `enganoso`, `distorcido`, `misleading`, `false`, `incorreto`, entre outros.

**True** - vereditos que contêm: `verdadeiro`, `correto`, `confirmado`, `verified`, `true`, entre outros.

Registros com vereditos ambíguos ou sem classificação clara foram **descartados** para garantir a qualidade do treino.

### Distribuição do dataset

| Label | Quantidade | Percentual |
|-------|-----------|------------|
| Fake  | 379 | 99,2% |
| True  | 3   | 0,8%  |

> ⚠️ **Sobre o desbalanceamento:** A proporção reflete a realidade das fontes — agências de fact-checking existem predominantemente para desmentir fake news, resultando em poucos registros com veredito verdadeiro.

##  Algoritmo Utilizado

> _A ser preenchido._

---

##  Como Foi Realizado o Treinamento

> _A ser preenchido._

---

##  Métricas de Avaliação

> _A ser preenchido._

---
