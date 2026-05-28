import argparse
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = URL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "texto" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV precisa ter as colunas: texto,label")
    df = df[["texto", "label"]].dropna()
    df["texto"] = df["texto"].astype(str).apply(clean_text)
    df = df[df["texto"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    return df


def compute_class_weights(labels, num_labels: int = 2) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_labels)
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


def split_dataframe(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    return train_df, val_df, test_df


def build_datasets(train_df, val_df, test_df):
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    return train_ds, val_ds, test_ds


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_batch(batch):
        return tokenizer(
            batch["texto"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["texto"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    return tokenized


def build_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Treino", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        if class_weights is None:
            loss = outputs.loss
        else:
            loss = F.cross_entropy(
                outputs.logits, batch["labels"], weight=class_weights
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def eval_epoch(model, dataloader, device, class_weights=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Validacao", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            if class_weights is None:
                loss = outputs.loss
            else:
                loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds)
            total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def predict_labels(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Teste", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist()
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_preds.extend(preds)

    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Treinamento BERTimbau para fake news")
    parser.add_argument("--data", default="dataset/treino.csv")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--output_dir", default="model")
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Desativa pesos de classe (padrao: ativado).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    print("=" * 40)
    print("INICIANDO TREINAMENTO")
    print("=" * 40)
    print(f"Dispositivo utilizado: {device_name}")

    df = load_dataframe(args.data)
    train_df, val_df, test_df = split_dataframe(df, args.seed)

    print("")
    print(f"Dataset carregado: {len(df)} exemplos")
    print(f"Treino: {len(train_df)}")
    print(f"Validacao: {len(val_df)}")
    print(f"Teste: {len(test_df)}")
    train_counts = train_df["label"].value_counts().to_dict()
    fake_count = train_counts.get(0, 0)
    true_count = train_counts.get(1, 0)
    print(f"Distribuicao no treino: FALSO={fake_count} | VERDADEIRO={true_count}")
    print("")
    print(f"Modelo: {args.model_name}")
    print(f"Epochs: {args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds, val_ds, test_ds = build_datasets(train_df, val_df, test_df)
    train_ds = tokenize_dataset(train_ds, tokenizer, args.max_length)
    val_ds = tokenize_dataset(val_ds, tokenizer, args.max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, args.max_length)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, args.batch_size
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "FALSO", 1: "VERDADEIRO"},
        label2id={"FALSO": 0, "VERDADEIRO": 1},
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config
    )
    model.to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_df["label"].values, num_labels=2)
        class_weights = class_weights.to(device)
        print(
            "Pesos de classe: "
            f"FALSO={class_weights[0].item():.2f} | "
            f"VERDADEIRO={class_weights[1].item():.2f}"
        )

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(args.outputs_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        print("\n" + "-" * 40)
        print(f"EPOCA {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights
        )
        val_loss, val_acc = eval_epoch(model, val_loader, device, class_weights)

        print(f"Loss treino: {train_loss:.4f}")
        print(f"Loss validacao: {val_loss:.4f}")
        print(f"Accuracy validacao: {val_acc * 100:.2f}%")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping: sem melhora na validacao.")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("\n" + "-" * 40)
    print("TREINAMENTO FINALIZADO")
    print("-" * 40)

    labels, preds = predict_labels(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels,
        preds,
        target_names=["FALSO", "VERDADEIRO"],
        digits=4,
        zero_division=0,
    )

    print(f"Accuracy final: {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print("")
    print("Confusion Matrix:")
    print(cm)
    print("")
    print("Relatorio por classe:")
    print(report)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("")
    print(f"Modelo salvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
