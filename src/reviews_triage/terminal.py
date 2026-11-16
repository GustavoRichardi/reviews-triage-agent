"""Peças comuns às ferramentas de terminal (revisão de rótulos e notas do juiz)."""

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd


class Quit(Exception):
    """O usuário digitou `q` (ou Ctrl+C / Ctrl+D)."""


class Back(Exception):
    """O usuário digitou `v`: voltar uma avaliação."""


def ask_command(ask: Callable[[str], str], prompt: str) -> str:
    """Lê uma resposta e transforma `q` e `v` em exceções, para o laço tratar uma vez só."""
    try:
        answer = ask(prompt)
    except (EOFError, KeyboardInterrupt):
        raise Quit from None
    command = answer.strip().lower()
    if command == "q":
        raise Quit
    if command == "v":
        raise Back
    return answer


def write_csv_atomic(rows: list[dict], columns: list[str], path: Path) -> None:
    """Grava num temporário e troca de uma vez: um Ctrl+C no meio não corrompe o arquivo."""
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
