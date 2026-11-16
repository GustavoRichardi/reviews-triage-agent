"""RAG sobre a política de atendimento: leitura dos .md, índice vetorial e busca.

Uso (para ver o que a busca devolve para um texto):
    python -m reviews_triage.policy "meu pedido não chegou" -k 3   # (ou `make policy-search`)
"""

import argparse
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import chromadb
from chromadb.config import Settings

POLICIES_DIR = Path("docs/politicas")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# O MiniLM multilíngue corta o texto em 128 tokens (~85 palavras em português). O que passar
# disso não entra no vetor. Cada seção da política precisa caber, e um teste confere.
MAX_CHUNK_WORDS = 80

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class PolicyChunk:
    source: str  # nome do arquivo, ex.: "atraso_entrega.md"
    title: str  # título do documento (H1)
    section: str  # título da seção (H2)
    text: str  # corpo da seção
    score: float = 0.0  # similaridade de cosseno com a consulta (1 = idêntico)

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.section}"

    @property
    def embedding_text(self) -> str:
        """O que vai para o vetor: título e seção dão contexto a um corpo curto."""
        return f"{self.title}. {self.section}. {self.text}"


def load_chunks(directory: Path = POLICIES_DIR) -> list[PolicyChunk]:
    """Um chunk por seção `##`. O texto antes da primeira seção (o aviso de política
    fictícia) fica de fora: não é conteúdo de política."""
    chunks: list[PolicyChunk] = []
    for path in sorted(directory.glob("*.md")):
        title, section, body = path.stem, None, []
        sections: list[tuple[str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if section:
                    sections.append((section, " ".join(body)))
                section, body = line[3:].strip(), []
            elif section and line.strip():
                body.append(line.strip())
        if section:
            sections.append((section, " ".join(body)))
        if not sections:
            raise ValueError(f"{path} não tem nenhuma seção '## ...'")
        chunks += [PolicyChunk(path.name, title, name, text) for name, text in sections]
    if not chunks:
        raise FileNotFoundError(f"Nenhuma política em {directory}")
    return chunks


class SentenceTransformerEmbedder:
    """Embeddings locais (sem custo de API, sem enviar texto a terceiros)."""

    def __init__(self, model_name: str | None = None):
        # Import aqui: carregar o torch leva segundos e não queremos pagar isso em
        # testes, `--help` ou qualquer caminho que não use embeddings reais.
        from sentence_transformers import SentenceTransformer

        name = model_name or os.environ.get("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        self._model = SentenceTransformer(name)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        # Vetores normalizados: o produto interno vira o cosseno.
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class PolicyIndex:
    """Índice vetorial (Chroma em memória) das seções da política."""

    def __init__(self, chunks: list[PolicyChunk], embed: Embedder):
        self._chunks = chunks
        self._embed = embed
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self._collection = client.create_collection(
            name=f"policies_{uuid.uuid4().hex}",  # único: o cliente em memória é compartilhado
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # os vetores vêm de `embed`; nada de modelo padrão do Chroma
        )
        self._collection.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=embed([c.embedding_text for c in chunks]),
            documents=[c.text for c in chunks],
        )

    def search(self, query: str, k: int = 3) -> list[PolicyChunk]:
        result = self._collection.query(
            query_embeddings=self._embed([query]), n_results=min(k, len(self._chunks))
        )
        return [
            replace(self._chunks[int(chunk_id)], score=1.0 - distance)
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]


def build_index(directory: Path = POLICIES_DIR, embed: Embedder | None = None) -> PolicyIndex:
    return PolicyIndex(load_chunks(directory), embed or SentenceTransformerEmbedder())


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca trechos da política de atendimento.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    for chunk in build_index().search(args.query, args.k):
        print(f"{chunk.score:.3f}  {chunk.ref}\n       {chunk.text}\n")


if __name__ == "__main__":
    main()
