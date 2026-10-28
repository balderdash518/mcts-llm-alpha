"""Transport contract tests: streaming failures must not masquerade as alpha evidence."""
from types import SimpleNamespace as NS
import json
import pytest

from mcts_llm_alpha.config.config import LLMConfig
from mcts_llm_alpha.llm.client import LLMClient, GenerationError, parse_json
from mcts_llm_alpha.llm.context import estimate_tokens, fit_levels
from mcts_llm_alpha.llm.transport import CompletionTransport, CompletionError


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False
    def __iter__(self):
        yield from self.chunks
    def close(self):
        self.closed = True


def chunk(text=None, reasoning=None, finish=None, usage=None):
    return NS(choices=[NS(index=0, delta=NS(content=text, reasoning_content=reasoning),
                          finish_reason=finish)] if usage is None else [], usage=usage)


def transport(**options):
    return CompletionTransport('test-only', 'test-model', langfuse='off', retry_delay=0, **options)


def test_brief_is_rerendered_coarser_until_it_fits():
    levels = []
    def build(level):
        levels.append(level)
        return 'x' * (12000 // (level + 1))
    prompt, level = fit_levels(build, 6000, 2048)
    assert level > 0 and levels == list(range(level + 1))
    assert estimate_tokens(prompt) + 2048 + 256 <= 6000
    with pytest.raises(ValueError, match='coarsest'):
        fit_levels(lambda level: 'x' * 100000, 6000, 2048)


def test_budget_is_dynamic_and_chinese_is_counted_conservatively():
    t = transport(extra_body={'chat_template_kwargs': {'enable_thinking': True}})
    short, meta = t.build_request('short')
    long, _ = t.build_request('长提示词' * 1000)
    assert short['max_tokens'] > 32000 > long['max_tokens']
    assert short['max_tokens'] + meta['estimated_prompt_tokens'] + 256 <= 32768
    assert short['extra_body']['chat_template_kwargs']['enable_thinking']
    assert short['stream_options'] == {'include_usage': True}
    assert 'reasoning_effort' not in short  # Unsupported provider knobs are not invented.
    t.close()


def test_mandatory_oversize_raises_without_calling_provider():
    t = transport(context_window=1500, min_output_tokens=512)
    with pytest.raises(ValueError, match='Prompt exceeds'):
        t.complete('中文' * 4000)
    t.close()


def test_stream_separates_thinking_and_consumes_usage_only_tail():
    t = transport()
    usage = NS(model_dump=lambda **kw: {'prompt_tokens': 10, 'completion_tokens': 90, 'total_tokens': 100})
    stream = Stream([chunk(reasoning='provider reasoning'), chunk('<think>inline'),
                     chunk(' reasoning</think>{"formula":"Mean($close,5)"}'),
                     chunk(finish='stop'), chunk(usage=usage)])
    text, meta = t._consume(stream)
    assert text == '{"formula":"Mean($close,5)"}'
    assert meta['usage']['total_tokens'] == 100 and meta['reasoning_characters'] > 0
    assert stream.closed
    t.close()


def test_interrupted_stream_is_discarded_before_retry():
    t = transport(max_retries=2)
    streams = [Stream([chunk('BROKEN')]), Stream([chunk('COMPLETE', finish='stop')])]
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return streams[len(calls)-1]
    t.client.chat.completions.create = create
    assert t.complete('test') == 'COMPLETE'
    assert len(calls) == 2 and all(s.closed for s in streams)
    t.close()


def test_length_limited_answer_is_rejected_without_duplicate_paid_retries():
    t = transport(max_retries=3)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return Stream([chunk('{"formula":', finish='length')])
    t.client.chat.completions.create = create
    with pytest.raises(CompletionError, match='length'):
        t.complete('test')
    assert len(calls) == 1
    t.close()


def test_auth_error_is_not_retried_or_leaked():
    t = transport(max_retries=3)
    calls = []
    class AuthError(Exception):
        status_code = 401
    def create(**kwargs):
        calls.append(kwargs)
        raise AuthError('secret-test-key')
    t.client.chat.completions.create = create
    with pytest.raises(CompletionError) as error:
        t.complete('test')
    assert 'secret-test-key' not in str(error.value) and len(calls) == 1
    t.close()


def test_non_stream_and_legacy_sync_use_same_transport():
    t = transport(stream=False)
    response = NS(choices=[NS(index=0, message=NS(content='<think>reason</think>42'),
                              finish_reason='stop')], usage=None)
    t.client.chat.completions.create = lambda **kw: response
    client = object.__new__(LLMClient)
    client.transport = t
    assert client.get_completion_sync('test', max_tokens=1000) == '42'
    t.close()


def test_custom_endpoint_cannot_inherit_openai_credential(monkeypatch):
    with pytest.raises(ValueError, match='own api_key_env'):
        LLMConfig(base_url='https://neuroii.com/v1')
    config = LLMConfig(base_url='https://neuroii.com/v1', api_key_env='NEUROII_API_KEY')
    monkeypatch.delenv('NEUROII_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-forward')
    with pytest.raises(ValueError, match='NEUROII_API_KEY'):
        LLMClient.from_config(config)

def scripted_client(answers):
    client = object.__new__(LLMClient)
    client.temperatures = {'review': 0.1, 'formula': 1.0}
    client.json_retries = 2
    calls = []
    def complete(prompt, temperature=0.7, max_tokens=None, name='', role=None):
        calls.append((temperature, name))
        return answers.pop(0)
    client.get_completion_sync = complete
    return client, calls


def test_review_parses_fenced_json_and_uses_paper_temperature():
    client, calls = scripted_client(['```json\n{"score": 7.5, "reason": "ok"}\n```'])
    assert client.review_overfitting('Ma($close, 5)', ['h'], []) == (7.5, 'ok')
    assert calls == [(0.1, 'overfitting-review')]


def test_unparseable_answers_raise_instead_of_defaulting():
    client, calls = scripted_client(['not json', '{"score": 11, "reason": "x"}', '{"reason": "x"}'])
    with pytest.raises(GenerationError):
        client.review_overfitting('Ma($close, 5)', [], [])
    assert len(calls) == 3
    client, _ = scripted_client(['{"formula": "", "parameters": {}, "candidates": []}'] * 3)
    with pytest.raises(GenerationError):
        client.formula({'name': 'a', 'description': 'b', 'pseudo_code': ['c']}, ('close',), [])
    assert parse_json('前言 {"a": 1} 后记') == {'a': 1}


def test_length_limited_answer_becomes_a_node_level_generation_failure():
    client, _ = scripted_client([])
    def truncated(prompt, temperature=0.7, max_tokens=None, name='', role=None):
        raise CompletionError('Completion ended with length; no complete answer accepted.')
    client.get_completion_sync = truncated
    with pytest.raises(GenerationError, match='portrait-to-formula: Completion ended with length'):
        client.formula({'name': 'a', 'description': 'b', 'pseudo_code': ['c']}, ('close',), [])
