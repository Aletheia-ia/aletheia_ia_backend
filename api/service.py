import os
import torch

from transformers import AutoTokenizer

from api.config import DEFAULT_THRESHOLD, MAX_LENGTH
from api.schemas import PredictResponse

from lib.config import DATASET_PATH, MODEL_DIR
from lib.predict import clean_text, load_model, run_inference


class ModelService:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        model_source = MODEL_DIR
        if os.path.isdir(model_source):
            pass
        elif "/" in model_source:
            # Repositório no Hugging Face Hub (ex.: usuario/aletheia-bert)
            pass
        else:
            raise FileNotFoundError(
                f"Modelo não encontrado em '{model_source}'. "
                f"Treine localmente (python train.py --data {DATASET_PATH}) "
                "ou defina HF_MODEL_ID com o repositório no Hub."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
        self.model = load_model(model_source, torch_dtype=None)
        self.model.to(self.device)
        self.model.eval()

    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def get_status(self) -> dict:
        return {
            "status": "ok" if self.is_ready() else "loading",
            "model_loaded": self.is_ready(),
            "device": self.device.type,
        }

    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> PredictResponse:
        if not self.is_ready():
            raise RuntimeError("Modelo ainda não foi carregado.")

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Texto inválido ou vazio após limpeza.")

        label, confidence, prob_fake, prob_true = run_inference(
            self.model,
            self.tokenizer,
            cleaned,
            self.device,
            MAX_LENGTH,
            threshold,
        )

        return PredictResponse(
            texto=cleaned,
            label=label,
            confianca=round(confidence, 1),
            prob_falso=round(prob_fake, 1),
            prob_verdadeiro=round(prob_true, 1),
            limiar=threshold,
        )


model_service = ModelService()

