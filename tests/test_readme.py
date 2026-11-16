from pathlib import Path

import pytest

from reviews_triage.readme import render_judge_block, update_readme


def test_update_readme_replaces_only_the_marked_block(tmp_path: Path):
    readme = tmp_path / "README.md"
    readme.write_text("antes\n<!-- auto:x -->\nvelho\n<!-- /auto:x -->\ndepois\n", encoding="utf-8")

    update_readme({"x": "novo"}, readme)
    update_readme({"x": "mais novo"}, readme)  # idempotente

    assert readme.read_text(encoding="utf-8") == (
        "antes\n<!-- auto:x -->\nmais novo\n<!-- /auto:x -->\ndepois\n"
    )
    with pytest.raises(ValueError, match="auto:y"):
        update_readme({"y": "z"}, readme)


def test_judge_block_publishes_the_kappa_not_only_the_verdict():
    result = {
        "n": 30,
        "criterios": {
            "aderencia": {
                "kappa_quadratico": 0.612,
                "ate_1_ponto": 0.9,
                "vies_juiz_menos_humano": 0.2,
                "veredito": "aprovado",
            },
            "tom": {
                "kappa_quadratico": None,
                "ate_1_ponto": 1.0,
                "vies_juiz_menos_humano": 0.0,
                "veredito": "inconclusivo",
            },
        },
    }
    block = render_judge_block(result, "run1")
    assert "**0.61**" in block and "aprovado" in block and "inconclusivo" in block


def test_real_readme_has_the_auto_blocks():
    text = Path("README.md").read_text(encoding="utf-8")
    for name in ("rotulagem", "validacao-juiz"):
        assert f"<!-- auto:{name} -->" in text and f"<!-- /auto:{name} -->" in text
