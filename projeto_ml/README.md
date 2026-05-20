# Projeto ML - Fake News Eleitorais (BERTimbau)

Sistema completo de Machine Learning para classificar afirmacoes eleitorais brasileiras como falsas (0) ou verdadeiras (1), usando BERTimbau com PyTorch e Hugging Face Transformers.

## Estrutura

```
projeto_ml/
|-- dataset/
|   `-- treino.csv
|-- model/
|-- outputs/
|-- train.py
|-- predict.py
|-- requirements.txt
`-- README.md
```

## Dataset

Formato esperado:

```csv
texto,label
"As urnas foram fraudadas",0
"O Brasil realizou eleicoes em 2022",1
```

Colunas:
- `texto`: afirmacao textual
- `label`: 0 (fake) ou 1 (verdadeiro)

## Instalacao

```bash
pip install -r requirements.txt
```

## Treinamento

```bash
python train.py --data dataset/treino.csv
```

Parametros principais (com padrao definido no codigo):
- `--epochs` (3)
- `--batch_size` (8)
- `--learning_rate` (2e-5)
- `--weight_decay` (0.01)
- `--max_length` (128)
- `--patience` (2)
- `--no_class_weights` (desativa pesos de classe)

O script:
- limpa texto (URLs, emojis, espacos duplicados)
- faz split 70/15/15 com stratify
- treina com early stopping
- exibe metricas completas no terminal
- salva modelo e tokenizer em `model/`
- usa pesos de classe por padrao (melhor para datasets desbalanceados)

## Inferencia

```bash
python predict.py --text "As urnas foram fraudadas"
```

Modo interativo (mantem o modelo carregado para multiplas consultas):

```bash
python predict.py --interactive
```

Ajuste de limiar para classificar como verdadeiro (ex: 0.60):

```bash
python predict.py --text "..." --threshold 0.60
```

## Feedback manual

Se o modelo errar, você pode auditá-lo manualmente com o script interativo:

```bash
python interactive_feedback.py
```

**Fluxo:**
1. Digite uma notícia/afirmação
2. Informe o label correto: `0` (falso) ou `1` (verdadeiro)
3. O modelo fará uma predição e exibirá a confiança
4. Se o modelo errar, o exemplo é **salvo automaticamente** no dataset
5. Digite `sair` para encerrar

**Resumo ao final:**
- Total de textos avaliados
- Acertos e erros (com percentual)
- Lista de todos os erros do modelo
- Quantidade de novos exemplos salvos

**Re-treinar com feedback:**

```bash
python train.py --data dataset/treino.csv
```

Este script mantém o modelo carregado para múltiplas avaliações consecutivas (mais rápido).

GPU com float16 (mais rapido):

```
Texto:
"As urnas foram fraudadas"

Resultado:
FALSO

Confianca:
93.2%

Probabilidades:
FALSO: 93.2% | VERDADEIRO: 6.8%
Limiar VERDADEIRO: 0.50
```

## Saidas

- Modelo e tokenizer: `model/`
- Melhor checkpoint de treino: `outputs/best_model.pt`

## GPU

O sistema detecta CUDA automaticamente e usa GPU se disponivel.
