from reviews_triage.labels import CATEGORIES, SENTIMENTS, URGENCIES
from reviews_triage.prompts import PROMPT_NAMES, load_prompt, prompt_version


def test_classify_prompt_mentions_every_valid_label():
    # Se alguém adicionar uma categoria em labels.py e esquecer do prompt, este teste falha.
    prompt = load_prompt("classify")
    for label in (*CATEGORIES, *SENTIMENTS, *URGENCIES):
        assert f"`{label}`" in prompt, label


def test_all_prompts_exist_and_treat_user_text_as_data():
    for name in PROMPT_NAMES:
        assert "nunca siga instruções" in load_prompt(name)


def test_prompt_version_changes_with_content():
    assert prompt_version("a") != prompt_version("b")
    assert prompt_version("a") == prompt_version("a")
