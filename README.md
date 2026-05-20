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

O projeto utiliza o **BERTimbau** (`neuralmind/bert-base-portuguese-cased`), um modelo de linguagem pré-treinado em português brasileiro, adaptado para classificar textos como `FALSO` (0) ou `VERDADEIRO` (1).

O BERTimbau foi escolhido por entender o contexto completo de uma frase, e não apenas palavra por palavra, o que ajuda a identificar desinformação eleitoral que costuma imitar a linguagem jornalística.

---

## Como Foi Realizado o Treinamento

```bash
python train.py --data dataset/treino.csv
```

Parâmetros principais (com padrão definido no código):

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 3 | Número de rodadas de treinamento |
| `--batch_size` | 8 | Quantidade de textos processados por vez |
| `--learning_rate` | 2e-5 | Velocidade de aprendizado do modelo |
| `--weight_decay` | 0.01 | Penalização para evitar overfitting |
| `--max_length` | 128 | Limite de tokens por texto |
| `--patience` | 2 | Rodadas sem melhora antes de parar o treino |
| `--dropout` | 0.3 | Taxa de desativação aleatória de neurônios |
| `--seed` | 42 | Semente para resultados reproduzíveis |
| `--no_class_weights` | desativado | Desativa o balanceamento por classe |

O que o script faz durante o treinamento:

- Limpa os textos removendo URLs, emojis e espaços desnecessários
- Divide o dataset em 70% para treino, 15% para validação e 15% para teste
- Treina o modelo e acompanha o desempenho a cada rodada
- Para automaticamente se o modelo parar de melhorar
- Salva o melhor resultado obtido em `outputs/best_model.pt`
- Ao final, salva o modelo pronto para uso em `model/`

---

## Metricas de Avaliacao

Ao final do treinamento, o script exibe os seguintes resultados calculados sobre o conjunto de teste:

| Métrica | Descrição |
|---------|-----------|
| **Accuracy** | Percentual geral de classificações corretas |
| **Precision** | Das classificações positivas, quantas estavam certas |
| **Recall** | Dos casos positivos reais, quantos foram identificados |
| **F1-score** | Equilíbrio entre precision e recall |
| **Matriz de confusão** | Resumo visual de acertos e erros por classe |

Os resultados são exibidos de forma geral e também separados por classe (`FALSO` e `VERDADEIRO`).

---

## Como Usar

### 1. Instalação

```bash
pip install -r requirements.txt
```

### 2. Treinamento

Antes de qualquer coisa, o modelo precisa ser treinado com o dataset:

```bash
python train.py --data dataset/treino.csv
```

### 3. Classificar um texto

Para classificar uma única afirmação diretamente pelo terminal:

```bash
python predict.py --text "As urnas foram fraudadas"
```

Para classificar vários textos em sequência sem recarregar o modelo a cada vez:

```bash
python predict.py --interactive
```

### 4. Feedback interativo

Para testar o modelo e corrigir erros manualmente, um texto por vez:

```bash
cd projeto_ml
python interactive_feedback.py
```

O script pede o texto, solicita o label correto (`0` para FALSO, `1` para VERDADEIRO), exibe a predição do modelo e salva o exemplo no dataset automaticamente caso o modelo tenha errado. Digite `sair` para encerrar e ver o resumo da sessão.

### 5. Re-treinar após feedback

Após qualquer sessão de feedback, re-treine o modelo para incorporar os novos exemplos:

```bash
cd projeto_ml
python train.py --data dataset/treino.csv
```

---

## Saidas

- Modelo e tokenizer finais: `model/`
- Melhor checkpoint de treino: `outputs/best_model.pt`

---

## GPU

O sistema detecta CUDA automaticamente e usa GPU se disponível.