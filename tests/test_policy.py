from pathlib import Path

import pytest
from fakes import hash_embed

from reviews_triage.policy import MAX_CHUNK_WORDS, PolicyIndex, build_index, load_chunks

POLICIES = Path("docs/politicas")


def test_every_policy_file_declares_it_is_fictional():
    # O aviso fica no topo de cada arquivo (acima da primeira seção).
    for path in POLICIES.glob("*.md"):
        head = path.read_text(encoding="utf-8").split("\n## ")[0]
        assert "Documento fictício" in head, f"{path.name} sem o aviso de política fictícia"


def test_chunks_follow_the_section_structure_and_skip_the_banner():
    chunks = load_chunks(POLICIES)

    assert len(chunks) >= 15
    assert all(c.section and c.text for c in chunks)
    assert not any("fictício" in c.text for c in chunks)
    atraso = [c for c in chunks if c.source == "atraso_entrega.md"]
    assert {c.section for c in atraso} >= {
        "Atraso de 3 a 7 dias úteis",
        "Atraso acima de 7 dias úteis",
    }


def test_every_chunk_fits_the_embedding_model_window():
    # Acima disso o MiniLM trunca em silêncio e o fim da seção some do vetor.
    for chunk in load_chunks(POLICIES):
        words = len(chunk.embedding_text.split())
        assert words <= MAX_CHUNK_WORDS, f"{chunk.ref} tem {words} palavras"


def test_policy_encodes_the_agreed_values():
    text = " ".join(c.text for c in load_chunks(POLICIES))

    for expected in [
        "3 dias úteis",
        "7 dias úteis",
        "15%",
        "10 dias úteis",
        "mesmo meio de pagamento",
    ]:
        assert expected in text


def test_search_returns_the_closest_chunk_first_with_scores():
    index = build_index(POLICIES, embed=hash_embed)
    chunks = load_chunks(POLICIES)

    for chunk in chunks:
        top = index.search(chunk.embedding_text, k=3)
        assert top[0].ref == chunk.ref  # buscar o próprio texto devolve a si mesmo
        assert top[0].score == pytest.approx(1.0, abs=1e-3)
        assert [r.score for r in top] == sorted((r.score for r in top), reverse=True)
        assert len(top) == 3


def test_search_k_is_capped_by_corpus_size():
    chunks = load_chunks(POLICIES)[:2]
    assert len(PolicyIndex(chunks, hash_embed).search("qualquer coisa", k=10)) == 2


def test_missing_policies_dir_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path)


def test_long_delay_compensations_are_a_choice_not_cumulative():
    # Acima de 7 dias úteis: cupom OU cancelamento; o frete é reembolsado nos dois casos.
    long_delay = next(
        c for c in load_chunks(POLICIES) if c.section == "Atraso acima de 7 dias úteis"
    )
    assert "escolhe" in long_delay.text and "não se somam" in long_delay.text
    assert "frete" in long_delay.text
