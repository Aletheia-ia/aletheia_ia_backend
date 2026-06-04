"""Arquivo de inicialização da API (executar a partir da raiz).

Uso:
  python start_api.py

Ou com parâmetros:
  python start_api.py --host 0.0.0.0 --port 8000 --reload

A API fica em api/main.py, expondo `app`.
"""

from __future__ import annotations

import argparse
import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iniciar API FastAPI - Aletheia IA")
    parser.add_argument("--host", default="0.0.0.0", help="Host do servidor (padrão: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor (padrão: 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Habilita reload automático (útil para desenvolvimento)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de workers do uvicorn (padrão: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import local para garantir que o projeto esteja carregado com o path correto
    # (o start_api.py roda a partir da raiz).
    from api.app import app

    # Defaults puxados das definições em /api (api/config.py)
    from api.config import HOST as CONFIG_HOST, PORT as CONFIG_PORT

    uvicorn.run(
        app,
        host=args.host if args.host is not None else CONFIG_HOST,
        port=args.port if args.port is not None else CONFIG_PORT,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

