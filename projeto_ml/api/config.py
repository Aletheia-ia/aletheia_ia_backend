import os

API_TITLE = "Aletheia IA API"
API_DESCRIPTION = "Classificador de fake news eleitorais em português"
API_VERSION = "1.0.0"

MODEL_DIR = os.getenv("MODEL_DIR", "model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))

# Em produção, substitua "*" pelo domínio do frontend
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
