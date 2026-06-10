import os

DATASET_PATH = "dataset/treino.csv"
DATASET_RAW_PATH = "dataset/dataset_raw.csv"

# Local: "model" | Hugging Face Hub: "seu-usuario/aletheia-bert"
MODEL_DIR = os.getenv("HF_MODEL_ID", os.getenv("MODEL_DIR", "model"))