import os
import time
import pandas as pd
from factcheckexplorer import FactCheckLib
from langdetect import detect

KEYWORDS = [
    "eleição", "Bolsonaro", "lula", "pt",
    "campanha", "urna", "voto", "fraude"
]

OUTPUT_CSV = "dataset_raw.csv"
all_results = []

for keyword in KEYWORDS:
    print(f"\nBuscando: '{keyword}'...")
    temp_file = f"_temp_{keyword}.csv"

    try:
        fact_check = FactCheckLib(
            query=keyword,
            language=None,
            num_results=200,
            csv_filename=temp_file
        )
        fact_check.process()

        if os.path.exists(temp_file):
            df_temp = pd.read_csv(temp_file)
            df_temp["palavra_chave"] = keyword
            all_results.append(df_temp)
            os.remove(temp_file)
            print(f"   {len(df_temp)} registros coletados")
        else:
            print(f"   Nenhum resultado para '{keyword}'")

    except Exception as e:
        print(f"   Erro em '{keyword}': {e}")

    time.sleep(2)

if not all_results:
    print("Nenhum dado coletado.")
    exit()

df_final = pd.concat(all_results, ignore_index=True)
df_final.drop_duplicates(subset=["Claim"], inplace=True)

# Filtro de idioma
def is_portuguese(texto):
    try:
        return detect(str(texto)) == "pt"
    except:
        return False

print("\nFiltrando apenas textos em português...")
df_final = df_final[df_final["Claim"].apply(is_portuguese)]
print(f"   {len(df_final)} registros em português")

# Rótulos - descarta tudo que não for claramente Fake ou True
TERMOS_FAKE = [
    "falso", "false", "fake", "incorreto", "incorrect",
    "enganoso", "misleading", "distorcido", "impreciso",
    "mentira", "infundado", "exagerado", "distorted",
    "unsupported", "pants on fire", "mostly false",
    "sem evidência", "não há provas", "desmentido",
    "satire", "satira", "enganoso:"
]
TERMOS_TRUE = [
    "verdadeiro", "true", "correto", "correct",
    "confirmado", "confirmed", "verified", "accurate",
    "mostly true", "legítimo", "procede", "verdade"
]

def padronizar_label(valor):
    if not isinstance(valor, str):
        return None
    v = valor.lower().strip()
    if any(p in v for p in TERMOS_FAKE):
        return "Fake"
    elif any(p in v for p in TERMOS_TRUE):
        return "True"
    return None   # descarta - dado incerto não entra no treino

df_final["label"] = df_final["Verdict"].apply(padronizar_label)
df_final = df_final[df_final["label"].notna()]   # remove os None

df_saida = pd.DataFrame({
    "texto":            df_final["Claim"],
    "label":            df_final["label"],
    "verdict_original": df_final["Verdict"],
    "fonte":            df_final["Source Name"],
    "url_original":     df_final["Source URL"],
    "data_checagem":    df_final["Review Publication Date"],
    "tags":             df_final["Tags"],
    "palavra_chave":    df_final["palavra_chave"],
})

df_saida = df_saida[df_saida["texto"].notna()]
df_saida.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

total = len(df_saida)
n_f   = (df_saida["label"] == "Fake").sum()
n_t   = (df_saida["label"] == "True").sum()

print(f"\n{'='*50}")
print(f"Dataset salvo: '{OUTPUT_CSV}'")
print(f"   Total:  {total}")
print(f"   Fake:   {n_f}  ({n_f/total*100:.1f}%)")
print(f"   True:   {n_t}  ({n_t/total*100:.1f}%)")
print(f"{'='*50}")