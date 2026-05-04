# coletar_true.py
import pandas as pd
import requests
import io

KEYWORDS = [
    "eleição", "bolsonaro", "lula", "pt",
    "campanha", "urna", "voto", "fraude"
]

# Carrega dataset atual

print("Carregando dataset atual...")
df_atual = pd.read_csv("dataset_raw.csv", encoding="utf-8-sig")
fake_atual = (df_atual["label"] == "Fake").sum()
true_atual = (df_atual["label"] == "True").sum()
print(f"   Fake: {fake_atual} | True: {true_atual}")

trues_necessarios = max(0, fake_atual - true_atual)
print(f"   Precisa de mais {trues_necessarios} True para equilibrar")


# Baixa Fake.br-Corpus

print("\nBaixando Fake.br-Corpus...")
URL = "https://raw.githubusercontent.com/roneysco/Fake.br-Corpus/master/preprocessed/pre-processed.csv"
response = requests.get(URL, timeout=30)
df_fakebr = pd.read_csv(io.StringIO(response.text))

# Filtra só as verdadeiras
df_true_br = df_fakebr[df_fakebr["label"] == "true"].copy()
print(f"   Notícias verdadeiras disponíveis: {len(df_true_br)}")

# Filtra por keyword
mask = df_true_br["preprocessed_news"].str.lower().str.contains(
    "|".join(KEYWORDS), na=False
)
df_true_filtrado = df_true_br[mask].copy()
print(f"   Com palavras-chave eleitorais: {len(df_true_filtrado)}")

# Se não tiver suficiente com keyword, completa sem filtro
if len(df_true_filtrado) < trues_necessarios:
    faltando = trues_necessarios - len(df_true_filtrado)
    print(f"   Insuficiente — completando com {faltando} sem filtro de keyword...")
    df_sem_filtro = df_true_br[~mask].head(faltando)
    df_true_filtrado = pd.concat([df_true_filtrado, df_sem_filtro], ignore_index=True)

# Limita ao necessário
df_true_filtrado = df_true_filtrado.head(trues_necessarios)

# Padroniza colunas
df_novos_true = pd.DataFrame({
    "texto":            df_true_filtrado["preprocessed_news"],
    "label":            "True",
    "verdict_original": "Notícia verdadeira — Fake.br-Corpus (PROPOR 2018)",
    "fonte":            "Fake.br-Corpus",
    "url_original":     "",
    "data_checagem":    "",
    "tags":             "",
    "palavra_chave":    "",
})

# Adiciona ao dataset atual e salva

df_final = pd.concat([df_atual, df_novos_true], ignore_index=True)
df_final.drop_duplicates(subset=["texto"], inplace=True)
df_final.to_csv("dataset_raw.csv", index=False, encoding="utf-8-sig")

total = len(df_final)
n_f   = (df_final["label"] == "Fake").sum()
n_t   = (df_final["label"] == "True").sum()

print(f"\n{'='*50}")
print(f"Dataset final salvo: 'dataset_raw.csv'")
print(f"   Total:  {total}")
print(f"   Fake:   {n_f}  ({n_f/total*100:.1f}%)")
print(f"   True:   {n_t}  ({n_t/total*100:.1f}%)")
print(f"{'='*50}")