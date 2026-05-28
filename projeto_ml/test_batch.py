"""Script rápido para testar batch de feedback."""
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer
from predict import clean_text, load_model, run_inference

MODEL_DIR = "model"
BATCH_FILE = "dataset/feedback_batch.csv"
MAX_LENGTH = 128
THRESHOLD = 0.5


def main():
    if not os.path.exists(MODEL_DIR):
        print(f"Modelo não encontrado em {MODEL_DIR}")
        exit(1)

    print("\n" + "="*60)
    print("TESTE EM LOTE - feedback_batch.csv")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = load_model(MODEL_DIR, torch_dtype=None)
    model.to(device)
    model.eval()

    df = pd.read_csv(BATCH_FILE)
    print(f"\nTotal de exemplos: {len(df)}")

    all_preds = []
    all_labels = []
    erros = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processando"):
        texto = str(row["texto"])
        label_real = int(row["label"])
        cleaned = clean_text(texto)

        if not cleaned:
            continue

        pred_label, _, _, _ = run_inference(
            model, tokenizer, cleaned, device, MAX_LENGTH, THRESHOLD
        )
        pred_num = 1 if pred_label == "VERDADEIRO" else 0

        all_preds.append(pred_num)
        all_labels.append(label_real)

        if pred_num != label_real:
            erros.append({
                "texto": cleaned[:60],
                "correto": label_real,
                "predito": pred_num
            })

    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nAcurácia: {acc*100:.2f}%")
    print(f"Acertos: {sum(1 for p,r in zip(all_preds, all_labels) if p==r)}/{len(all_labels)}")
    print(f"Erros: {len(erros)}/{len(all_labels)}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatriz de Confusão:")
    print(f"       Pred=0  Pred=1")
    print(f"Real=0  {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"Real=1  {cm[1,0]:3d}    {cm[1,1]:3d}")

    print("\nRelatório por classe:")
    print(classification_report(all_labels, all_preds,
                                target_names=["FALSO", "VERDADEIRO"],
                                digits=4, zero_division=0))

    print("="*60)


if __name__ == "__main__":
    main()
