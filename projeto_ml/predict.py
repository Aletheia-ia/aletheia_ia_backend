import argparse
import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

MODEL_DIR = "model"

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


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = URL_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_inference(
    model,
    tokenizer,
    text: str,
    device,
    max_length: int,
    threshold: float,
):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze(0)

    prob_fake = probs[0].item()
    prob_true = probs[1].item()
    label = "VERDADEIRO" if prob_true >= threshold else "FALSO"
    confidence = prob_true if label == "VERDADEIRO" else prob_fake
    return label, confidence * 100, prob_fake * 100, prob_true * 100


def load_model(model_dir: str, torch_dtype):
    kwargs = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype

    try:
        return AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    except TypeError:
        if torch_dtype is None:
            return AutoModelForSequenceClassification.from_pretrained(model_dir)
        return AutoModelForSequenceClassification.from_pretrained(
            model_dir, torch_dtype=torch_dtype
        )


def main():
    parser = argparse.ArgumentParser(description="Inferencia com BERTimbau")
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar para classificar como verdadeiro.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Mantem o modelo carregado para multiplas consultas.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Usa float16 na GPU para acelerar (pode reduzir levemente a precisao).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            "Diretorio do modelo nao encontrado. Execute o treino primeiro."
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("O limiar deve estar entre 0.0 e 1.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"

    print(f"Dispositivo utilizado: {device_name}")

    torch_dtype = None
    if args.fp16 and device.type == "cuda":
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = load_model(args.model_dir, torch_dtype)
    model.to(device)
    model.eval()

    def process_text(raw_text: str):
        cleaned = clean_text(raw_text)
        label, confidence, prob_fake, prob_true = run_inference(
            model,
            tokenizer,
            cleaned,
            device,
            args.max_length,
            args.threshold,
        )
        print("")
        print("Texto:")
        print(f'"{cleaned}"')
        print("")
        print("Resultado:")
        print(label)
        print("")
        print("Confianca:")
        print(f"{confidence:.1f}%")
        print("")
        print("Probabilidades:")
        print(f"FALSO: {prob_fake:.1f}% | VERDADEIRO: {prob_true:.1f}%")
        print(f"Limiar VERDADEIRO: {args.threshold:.2f}")

    text = args.text.strip()
    if text:
        process_text(text)
        return

    if args.interactive:
        print("\nModo interativo. Digite 'sair' para encerrar.")
        while True:
            user_text = input("Texto: ").strip()
            if user_text.lower() == "sair":
                break
            if not user_text:
                print("  ⚠ Texto não pode estar vazio.")
                continue
            process_text(user_text)
    else:
        user_text = input("Texto: ").strip()
        if user_text:
            process_text(user_text)


if __name__ == "__main__":
    main()
