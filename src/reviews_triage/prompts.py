"""Carrega os prompts versionados de prompts/*.md."""

import hashlib
from pathlib import Path

PROMPTS_DIR = Path("prompts")
PROMPT_NAMES = ("classify", "draft", "review")


def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> str:
    return (directory / f"{name}.md").read_text(encoding="utf-8").strip()


def prompt_version(text: str) -> str:
    """Hash curto do conteúdo. Muda sozinho quando o prompt muda, então não dá para
    esquecer de "subir a versão"; os relatórios da Fase 5 registram este valor."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
