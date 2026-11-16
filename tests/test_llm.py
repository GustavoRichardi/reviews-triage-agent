from types import SimpleNamespace

import pytest

from reviews_triage.llm import AnthropicLLM, LLMRefusal, LLMTruncated
from reviews_triage.schemas import Classification

PARSED = Classification(categoria="elogio", sentimento="positivo", urgencia="baixa")


def response(stop_reason="end_turn", text="ok", parsed=PARSED, details=None):
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=30,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    return SimpleNamespace(
        stop_reason=stop_reason,
        usage=usage,
        parsed_output=parsed,
        stop_details=details,
        content=[
            SimpleNamespace(type="thinking", text=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


class FakeClient:
    def __init__(self, resp):
        self.kwargs = None
        self.messages = SimpleNamespace(parse=self._call, create=self._call)
        self._resp = resp

    def _call(self, **kwargs):
        self.kwargs = kwargs
        return self._resp


def llm_with(resp, **kwargs):
    client = FakeClient(resp)
    return AnthropicLLM(model="m-test", client=client, **kwargs), client


def test_parse_sends_schema_effort_and_no_sampling_params():
    llm, client = llm_with(response(), effort="low")

    parsed, call = llm.parse("classify", "system", "user", Classification)

    assert parsed == PARSED
    assert client.kwargs["model"] == "m-test"
    assert client.kwargs["output_format"] is Classification
    assert client.kwargs["output_config"] == {"effort": "low"}
    assert client.kwargs["system"] == "system"
    assert "temperature" not in client.kwargs and "thinking" not in client.kwargs
    assert (call.step, call.input_tokens, call.output_tokens, call.cache_read_tokens) == (
        "classify",
        120,
        30,
        0,
    )
    assert call.latency_s >= 0


def test_text_returns_only_text_blocks_stripped():
    llm, _ = llm_with(response(text="  Olá!  "))
    assert llm.text("draft", "s", "u")[0] == "Olá!"


def test_refusal_is_an_explicit_error_carrying_the_cost():
    details = SimpleNamespace(category="general_harms")
    llm, _ = llm_with(response(stop_reason="refusal", details=details))

    with pytest.raises(LLMRefusal) as exc:
        llm.text("draft", "s", "u")

    assert exc.value.step == "draft" and exc.value.category == "general_harms"
    assert exc.value.call.input_tokens == 120


def test_truncated_output_is_an_error_not_a_silent_result():
    llm, _ = llm_with(response(stop_reason="max_tokens"))
    with pytest.raises(LLMTruncated):
        llm.parse("classify", "s", "u", Classification)


def test_missing_model_gives_a_clear_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_MODEL"):
        AnthropicLLM(client=object())


def test_model_and_effort_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "from-env")
    monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
    llm = AnthropicLLM(client=object())
    assert (llm.model, llm.effort) == ("from-env", "medium")
