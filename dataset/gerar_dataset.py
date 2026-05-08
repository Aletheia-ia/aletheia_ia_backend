import pandas as pd
import requests
import time
import re
from langdetect import detect
from dotenv import load_dotenv
import os

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

KEYWORDS = [
    "eleição", "Bolsonaro", "lula", "pt", "campanha", "fraude",
    "STF", "Sergio Moro", "pesquisa eleitoral", "candidato",
    "reeleição", "corrupção", "Dilma", "Michelle Bolsonaro",
    "Eduardo Bolsonaro", "Flavio Bolsonaro", "Ciro Gomes",
    "Haddad", "Temer", "Gleisi", "Datafolha",
    "golpe", "censura", "preso político",
    "delação", "mensalão", "propina", "ministro"
]

PALAVRAS_POLITICA = [
    "eleição", "voto", "urna", "bolsonaro", "lula", "pt", "campanha",
    "candidato", "presidente", "senado", "deputado", "governo",
    "político", "política", "congresso", "eleitor", "eleitoral",
    "partido", "coligação", "fraude", "democracia", "votação",
    "reeleição", "constituição",
    "stf", "tse", "tcu", "agu", "pgr", "cgu",
    "polícia federal", "ministério público", "tribunal",
    "câmara", "plenário", "inquérito", "habeas corpus",
    "posse", "mandato", "gestão", "orçamento",
    "projeto de lei", "medida provisória", "decreto",
    "oposição", "base aliada", "aliança",
    "delação", "investigado", "indiciado", "réu",
    "condenado", "preso", "corrupção",
    "mensalão", "petrolão", "lava jato",
    "pesquisa", "aprovação", "rejeição", "debate",
    "dilma", "temer", "moro", "haddad", "gleisi",
    "ministro", "golpe", "censura", "impeachment"
]


def tem_conteudo_politico(texto):
    if not isinstance(texto, str):
        return False
    texto_lower = texto.lower()
    matches = sum(1 for palavra in PALAVRAS_POLITICA if palavra in texto_lower)
    return matches >= 2


def coletar_fake():
    print("ETAPA 1: Coletando noticias falsas...")

    all_results = []

    for keyword in KEYWORDS:
        print(f"  '{keyword}'...", end=" ")
        temp_file = f"_temp_{keyword}.csv"

        try:
            from factcheckexplorer import FactCheckLib

            fact_check = FactCheckLib(
                query=keyword,
                language=None,
                num_results=400,
                csv_filename=temp_file
            )
            fact_check.process()

            if os.path.exists(temp_file):
                df_temp = pd.read_csv(temp_file)
                if df_temp is not None and len(df_temp) > 0:
                    df_temp["palavra_chave_query"] = keyword
                    all_results.append(df_temp)
                    print(f"{len(df_temp)} registros")
                else:
                    print("nenhum resultado")
                os.remove(temp_file)
            else:
                print("nenhum resultado")

        except Exception as e:
            print(f"erro: {str(e)[:80]}")

        time.sleep(2)

    all_results = [df for df in all_results if df is not None and len(df) > 0]
    if not all_results:
        print("Nenhuma noticia falsa coletada!")
        return None

    df_final = pd.concat(all_results, ignore_index=True)
    df_final.drop_duplicates(subset=["Claim"], inplace=True)

    def is_portuguese(texto):
        try:
            return detect(str(texto)) == "pt"
        except:
            return False

    print("Filtrando portugues...")
    df_final = df_final[df_final["Claim"].apply(is_portuguese)]
    print(f"  {len(df_final)} em portugues")

    TERMOS_FAKE = [
        "falso", "false", "fake", "incorreto", "incorrect",
        "enganoso", "misleading", "distorcido", "impreciso",
        "mentira", "infundado", "exagerado", "distorted",
        "unsupported", "pants on fire", "mostly false",
        "sem evidencia", "nao ha provas", "desmentido",
        "satire", "satira", "enganoso:"
    ]

    TERMOS_TRUE = [
        "verdadeiro", "true", "correto", "correct",
        "confirmado", "confirmed", "verified", "accurate",
        "mostly true", "legitimo", "procede", "verdade"
    ]

    def padronizar_label(valor):
        if not isinstance(valor, str):
            return None
        v = valor.lower().strip()
        if any(p in v for p in TERMOS_FAKE):
            return "Fake"
        elif any(p in v for p in TERMOS_TRUE):
            return "True"
        return None

    df_final["label"] = df_final["Verdict"].apply(padronizar_label)
    df_final = df_final[df_final["label"].notna()]

    df_saida = pd.DataFrame({
        "texto":            df_final["Claim"],
        "label":            df_final["label"],
        "verdict_original": df_final["Verdict"],
        "fonte":            df_final["Source Name"],
        "url_original":     df_final["Source URL"],
        "data_checagem":    df_final["Review Publication Date"],
        "tags":             df_final["Tags"],
        "palavra_chave":    None,
    })

    df_saida = df_saida[df_saida["texto"].notna()]
    df_fake = df_saida[df_saida["label"] == "Fake"].copy()

    def encontrar_keyword_no_texto(texto):
        if not isinstance(texto, str):
            return None
        texto_lower = texto.lower()
        for k in KEYWORDS:
            if re.search(r"\b" + re.escape(k.lower()) + r"\b", texto_lower):
                return k
        return None

    df_fake["palavra_chave"] = df_fake["texto"].apply(encontrar_keyword_no_texto)
    df_fake = df_fake[df_fake["palavra_chave"].notna()].copy()
    df_fake = df_fake[df_fake["texto"].apply(tem_conteudo_politico)].copy()
    print(f"  Fake relevantes: {len(df_fake)}")

    return df_fake


def coletar_true():
    print("\nETAPA 2: Coletando noticias verdadeiras...")

    noticias = []

    for keyword in KEYWORDS:
        print(f"  '{keyword}'...")

        try:
            for pagina in range(1, 4):
                params = {
                    'q':        keyword,
                    'language': 'pt',
                    'sortBy':   'relevancy',
                    'pageSize': 100,
                    'page':     pagina,
                    'apiKey':   NEWSAPI_KEY
                }

                response = requests.get(
                    'https://newsapi.org/v2/everything',
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                dados = response.json()

                if dados.get('status') != 'ok':
                    print(f"    p{pagina}: {dados.get('message', '')}")
                    break

                artigos = dados.get('articles', [])
                if not artigos:
                    break

                print(f"    pagina {pagina}: {len(artigos)} artigos")

                for artigo in artigos:
                    try:
                        titulo    = artigo.get('title', '')
                        descricao = artigo.get('description', '') or ''

                        if not titulo or len(titulo) < 20:
                            continue

                        texto_completo = (titulo + " " + descricao).lower()
                        kw_encontrada = None
                        for k in KEYWORDS:
                            if re.search(r"\b" + re.escape(k.lower()) + r"\b", texto_completo):
                                kw_encontrada = k
                                break

                        noticias.append({
                            'texto':            titulo,
                            'label':            'True',
                            'verdict_original': 'Noticia verdadeira - NewsAPI',
                            'fonte':            artigo.get('source', {}).get('name', 'NewsAPI'),
                            'url_original':     artigo.get('url', ''),
                            'data_checagem':    artigo.get('publishedAt', '')[:10],
                            'tags':             '',
                            'palavra_chave':    kw_encontrada,
                        })

                    except:
                        continue

                time.sleep(1)

        except Exception as e:
            print(f"    erro: {str(e)[:80]}")

    if not noticias:
        print("Nenhuma noticia verdadeira coletada!")
        return None

    df_true = pd.DataFrame(noticias)
    df_true.drop_duplicates(subset=['texto'], inplace=True)

    antes = len(df_true)
    df_true = df_true[df_true["texto"].apply(tem_conteudo_politico)].copy()
    print(f"  True relevantes: {len(df_true)} (removidas {antes - len(df_true)})")

    return df_true


def equilibrar_e_salvar(df_fake, df_true):
    print("\nETAPA 3: Equilibrando dataset...")

    if df_true is None or len(df_true) == 0:
        print("Nenhuma noticia verdadeira para equilibrar!")
        return

    def detectar_keyword(texto):
        if not isinstance(texto, str):
            return None
        texto_lower = texto.lower()
        for k in KEYWORDS:
            if re.search(r"\b" + re.escape(k.lower()) + r"\b", texto_lower):
                return k
        return None

    df_fake["palavra_chave"] = df_fake["texto"].apply(detectar_keyword)
    df_true["palavra_chave"] = df_true["texto"].apply(detectar_keyword)

    df_fake = df_fake[df_fake["palavra_chave"].notna()].copy()
    df_true = df_true[df_true["palavra_chave"].notna()].copy()

    fake_por_kw = df_fake["palavra_chave"].value_counts().to_dict()
    true_por_kw = df_true["palavra_chave"].value_counts().to_dict()

    lotes_fake = []
    lotes_true = []

    for kw in KEYWORDS:
        qtd_fake   = fake_por_kw.get(kw, 0)
        qtd_true   = true_por_kw.get(kw, 0)
        qtd_manter = min(qtd_fake, qtd_true)

        if qtd_manter == 0:
            print(f"  '{kw}': removido (sem cobertura em ambas as classes)")
            continue

        print(f"  '{kw}': {qtd_manter} (Fake:{qtd_fake} True:{qtd_true})")
        lotes_fake.append(df_fake[df_fake["palavra_chave"] == kw].head(qtd_manter))
        lotes_true.append(df_true[df_true["palavra_chave"] == kw].head(qtd_manter))

    if not lotes_fake or not lotes_true:
        print("Nenhuma keyword com cobertura em ambas as classes!")
        return

    df_fake_eq = pd.concat(lotes_fake, ignore_index=True)
    df_true_eq = pd.concat(lotes_true, ignore_index=True)
    df_final   = pd.concat([df_fake_eq, df_true_eq], ignore_index=True)
    df_final.drop_duplicates(subset=["texto"], inplace=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final.to_csv("dataset_raw.csv", index=False, encoding="utf-8-sig")

    total = len(df_final)
    n_f   = (df_final["label"] == "Fake").sum()
    n_t   = (df_final["label"] == "True").sum()

    print(f"\nDataset salvo: dataset_raw.csv")
    print(f"  Total: {total} | Fake: {n_f} ({n_f/total*100:.1f}%) | True: {n_t} ({n_t/total*100:.1f}%)")


if __name__ == "__main__":
    print("Pipeline iniciado\n")

    try:
        df_fake = coletar_fake()
        if df_fake is None:
            print("Falha ao coletar noticias falsas!")
            exit()

        df_true = coletar_true()
        if df_true is None:
            print("Falha ao coletar noticias verdadeiras!")
            exit()

        equilibrar_e_salvar(df_fake, df_true)

        print("\nPipeline concluido.")

    except Exception as e:
        print(f"\nErro fatal: {e}")
        import traceback
        traceback.print_exc()