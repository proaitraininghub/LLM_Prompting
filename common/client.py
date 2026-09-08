"""
Shared Azure OpenAI client setup used by every demo script in this repository.

All scripts import get_client(), get_chat_deployment(), and
get_embedding_deployment() from here so you only need to configure
credentials in one place: the .env file at the repo root.
"""
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

REQUIRED_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_CHAT_DEPLOYMENT"]


def _check_env():
    missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing) +
            "\nCopy .env.example to .env and fill in your Azure OpenAI / Foundry project details."
        )


def get_client() -> AzureOpenAI:
    _check_env()
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2026-01-01-preview"),
    )


def get_chat_deployment() -> str:
    _check_env()
    return os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]


def get_embedding_deployment() -> str:
    return os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
