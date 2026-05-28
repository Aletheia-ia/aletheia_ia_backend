"""
Script de Feedback Interativo para Classificação de Fake News Eleitorais.

Fluxo: Usuário digita texto → informa label correto (0/1) → modelo prediz →
resultado é exibido → salva no dataset apenas se o modelo errou.

Reutiliza funções do predict.py para evitar duplicação de código.
"""

import csv
import os

import torch
from transformers import AutoTokenizer

# Importar funções reutilizáveis do predict.py
from predict import clean_text, load_model, run_inference

# Configurações fixas do projeto
MODEL_DIR = "model"
DATASET_PATH = "dataset/treino.csv"
MAX_LENGTH = 128
THRESHOLD = 0.5


def append_to_dataset(cleaned_text: str, label: int) -> None:
    """Salva um exemplo no arquivo CSV do dataset de treino."""
    header_needed = not os.path.exists(DATASET_PATH)
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if header_needed:
            writer.writerow(["texto", "label"])
        writer.writerow([cleaned_text, label])


def get_user_label() -> int:
    """Solicita e valida o label correto do usuário (0 ou 1)."""
    while True:
        response = input("Label correto (0=FALSO, 1=VERDADEIRO): ").strip()
        if response in {"0", "1"}:
            return int(response)
        print("  ⚠ Resposta inválida. Digite 0 ou 1.")


def format_label(label: int) -> str:
    """Converte label numérico para string legível."""
    return "VERDADEIRO" if label == 1 else "FALSO"


def process_single_text(model, tokenizer, device, session_stats: dict) -> bool:
    """
    Processa um texto: solicita label, executa predição, compara resultados.
    
    Retorna False se usuário digitou 'sair', True caso contrário.
    """
    raw_text = input("\nTexto: ").strip()
    
    # Encerra o loop se usuário digitar 'sair'
    if raw_text.lower() == "sair":
        return False
    
    if not raw_text:
        print("  ⚠ Texto não pode estar vazio.")
        return True
    
    # Limpa o texto
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        print("  ⚠ Texto inválido após limpeza.")
        return True
    
    # Obtém label correto do usuário
    correct_label = get_user_label()
    
    # Executa predição do modelo
    if model is not None and tokenizer is not None:
        pred_label, confidence, prob_fake, prob_true = run_inference(
            model, tokenizer, cleaned_text, device, MAX_LENGTH, THRESHOLD
        )
        
        correct_label_str = format_label(correct_label)
        
        # Exibe resultado da predição
        print("\nResultado do Modelo:")
        print(f"   Predição: {pred_label}")
        print(f"   Confiança: {confidence:.1f}%")
        print(f"   (FALSO: {prob_fake:.1f}% | VERDADEIRO: {prob_true:.1f}%)")
        
        # Verifica se acertou
        model_was_correct = (pred_label == correct_label_str)
        if model_was_correct:
            print("   Acertou!")
            session_stats["acertos"] += 1
        else:
            print("   Errou!")
            session_stats["erros"] += 1
            session_stats["erros_detalhes"].append({
                "texto": cleaned_text[:80],
                "correto": correct_label_str,
                "predito": pred_label,
            })
            
            # Salva no dataset apenas se errou
            append_to_dataset(cleaned_text, correct_label)
            print(f"   Exemplo salvo no dataset.")
    
    session_stats["total"] += 1
    return True


def print_session_summary(stats: dict) -> None:
    """Exibe resumo da sessão com estatísticas e erros."""
    print("\n" + "=" * 70)
    print("RESUMO DA SESSÃO")
    print("=" * 70)
    
    total = stats["total"]
    if total == 0:
        print("Nenhum texto foi avaliado.")
        return
    
    acertos = stats["acertos"]
    erros = stats["erros"]
    taxa_acerto = (acertos / total) * 100 if total > 0 else 0
    
    print(f"\nTotal de textos avaliados: {total}")
    print(f"   Acertos: {acertos} ({taxa_acerto:.1f}%)")
    print(f"   Erros: {erros} ({100 - taxa_acerto:.1f}%)")
    
    if stats["erros_detalhes"]:
        print(f"\nTextos que o modelo errou:")
        for i, erro in enumerate(stats["erros_detalhes"], 1):
            print(f"\n   {i}. \"{erro['texto']}...\"")
            print(f"      Label correto: {erro['correto']}")
            print(f"      Modelo predisse: {erro['predito']}")
    
    print(f"\nExemplos salvos no dataset: {erros}")
    print("\n" + "=" * 70)
    if erros > 0:
        print("Para re-treinar o modelo com os novos exemplos, execute:")
        print("  python train.py --data dataset/treino.csv")
    print("=" * 70 + "\n")


def main():
    """Função principal: inicializa o loop interativo de feedback."""
    
    print("\n" + "=" * 70)
    print("FEEDBACK INTERATIVO - Classificação de Fake News Eleitorais")
    print("=" * 70)
    
    # Carrega modelo e tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isdir(MODEL_DIR):
        print(f"\nErro: Modelo não encontrado em '{MODEL_DIR}'")
        print("Execute: python train.py --data dataset/treino.csv")
        return
    
    # Carrega modelo e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = load_model(MODEL_DIR, torch_dtype=None)
    model.to(device)
    model.eval()
    
    print("Modelo carregado com sucesso.\n")
    
    # Inicializa estatísticas da sessão
    session_stats = {
        "total": 0,
        "acertos": 0,
        "erros": 0,
        "erros_detalhes": [],
    }
    
    # Loop interativo
    print("Instruções:")
    print("  - Digitar texto e pressionar ENTER")
    print("  - Informar o label correto (0 ou 1)")
    print("  - Modelo fará a predição e comparará com o label correto")
    print("  - Se errar, o exemplo será salvo automaticamente")
    print("  - Digite 'sair' para encerrar\n")
    
    while True:
        if not process_single_text(model, tokenizer, device, session_stats):
            break
    
    # Exibe resumo final
    print_session_summary(session_stats)


if __name__ == "__main__":
    main()
