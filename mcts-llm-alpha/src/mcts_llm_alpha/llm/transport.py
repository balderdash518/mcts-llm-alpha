"""OpenAI-compatible streaming transport with thinking and flexible token budgets."""
import json
import logging
import re
import time
from datetime import datetime, timezone

from .context import estimate_tokens, fit_prompt
from ..observability import SearchTracer

log = logging.getLogger(__name__)


def merge(base, override):
    """Recursive dict merge; ``override`` wins."""
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and \
            isinstance(result.get(key), dict) else value
    return result


class CompletionError(RuntimeError):
    """No complete final answer; partial text must not become a candidate."""


def split_thinking(text):
    if '</think>' in text and '<think>' not in text:
        return text.partition('</think>')[2].strip()
    return re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


class CompletionTransport:
    def __init__(self, api_key, model, max_retries=3, retry_delay=1.0, *,
                 base_url=None, timeout_s=600, stream=True, max_tokens=32768,
                 context_window=32768, min_output_tokens=4096, extra_body=None,
                 reasoning_effort=None, langfuse='auto', session_id=None):
        self.model, self.max_retries, self.retry_delay = model, max_retries, retry_delay
        self.base_url = base_url
        self.context_window, self.max_tokens = context_window, max_tokens
        self.min_output_tokens, self.stream = min_output_tokens, stream
        self.extra_body, self.reasoning_effort = extra_body or {}, reasoning_effort
        self.tracer = SearchTracer(langfuse, session_id)
        if self.tracer.client:
            from langfuse.openai import OpenAI
        else:
            from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)
        self.last_call = {}
        self.call_log_path = None   # optional JSONL copy of every call for offline review

    def _log_call(self, name, prompt, temperature, started, answer=None, error=None):
        if not self.call_log_path:
            return
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), "name": name,
               "temperature": temperature, "latency_s": round(time.monotonic() - started, 2),
               "prompt": prompt, "answer": answer, "error": error,
               **{k: self.last_call.get(k) for k in ("usage", "finish_reason", "attempt",
                                                      "estimated_prompt_tokens")}}
        with open(self.call_log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def build_request(self, prompt, temperature=.7, max_tokens=None, extra_body=None):
        if getattr(self, 'request_context', ''):
            prompt = self.request_context + '\n\n' + prompt
        requested = self.max_tokens if max_tokens is None else max_tokens
        if requested <= 0:
            raise ValueError('max_tokens must be positive')
        fitted, omitted = fit_prompt(prompt, self.context_window, min(self.min_output_tokens, requested))
        estimated = estimate_tokens(fitted)
        available = self.context_window - estimated - 256
        if available <= 0:
            raise ValueError('No output budget remains in model context')
        kwargs = dict(model=self.model, messages=[{'role': 'user', 'content': fitted}],
                      temperature=temperature, max_tokens=min(requested, available), stream=self.stream)
        if self.stream:
            kwargs['stream_options'] = {'include_usage': True}
        body = merge(self.extra_body, extra_body or {})
        if body:
            kwargs['extra_body'] = body
        if self.reasoning_effort:
            kwargs['reasoning_effort'] = self.reasoning_effort
        return kwargs, {'estimated_prompt_tokens': estimated, 'omitted_evidence_records': omitted,
                        'context_window': self.context_window, 'output_budget': kwargs['max_tokens']}

    def _consume(self, response):
        content, reasoning, usage, finish = [], [], {}, None
        started, progress = time.monotonic(), time.monotonic()
        try:
            for chunk in response if self.stream else [response]:
                if getattr(chunk, 'usage', None):
                    usage = chunk.usage.model_dump(exclude_none=True)
                for choice in chunk.choices or []:
                    if getattr(choice, 'index', 0) != 0:
                        continue
                    message = choice.delta if self.stream else choice.message
                    content.append(getattr(message, 'content', None) or '')
                    reasoning.append(getattr(message, 'reasoning_content', None)
                                     or getattr(message, 'reasoning', None) or '')
                    finish = choice.finish_reason or finish
                if time.monotonic() - progress >= 30:
                    log.info('LLM stream active: %.0fs', time.monotonic() - started)
                    progress = time.monotonic()
        finally:
            if self.stream:
                response.close()
        raw = ''.join(content)
        answer = split_thinking(raw)
        info = dict(usage=usage, finish_reason=finish, reasoning_characters=len(''.join(reasoning)),
                    inline_thinking=bool(re.search(r'</?think>', raw, re.IGNORECASE)))
        if finish is None:
            raise ConnectionError('Stream ended without finish_reason; partial output discarded')
        if finish != 'stop':
            raise CompletionError(f'Completion ended with {finish}; no complete answer accepted. '
                                  'Increase output budget or reduce prompt size if length-limited.')
        if not answer:
            raise CompletionError('Model returned no final answer')
        return answer, info

    def complete(self, prompt, temperature=.7, max_tokens=None, name='model-completion', extra_body=None):
        kwargs, budget = self.build_request(prompt, temperature, max_tokens, extra_body)
        started = time.monotonic()
        with self.tracer.observe(name + '-request', as_type='chain', input=kwargs['messages'],
                                 metadata=budget) as observation:
            for attempt in range(self.max_retries):
                self.last_call = {**budget, 'attempt': attempt + 1}
                try:
                    call = dict(kwargs)
                    if self.tracer.client:
                        call.update(name=name, metadata={'attempt': attempt + 1, **budget})
                    answer, info = self._consume(self.client.chat.completions.create(**call))
                    self.last_call.update(info)
                    observation.update(output=answer, metadata=self.last_call, level='DEFAULT', status_message='')
                    self._log_call(name, kwargs['messages'][0]['content'], temperature, started, answer)
                    return answer
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    retryable = (status in (408, 429) or (status is not None and status >= 500)
                                 or isinstance(exc, (ConnectionError, TimeoutError))
                                 or type(exc).__name__ in {'APIConnectionError', 'APITimeoutError'})
                    self.last_call.update(error_type=type(exc).__name__, status_code=status)
                    observation.update(level='ERROR', status_message=type(exc).__name__, metadata=self.last_call)
                    if not retryable or attempt + 1 == self.max_retries:
                        self._log_call(name, kwargs['messages'][0]['content'], temperature, started,
                                       error=f'{type(exc).__name__}, status={status}')
                        if isinstance(exc, CompletionError):
                            raise
                        # Do not expose a provider error body/headers; they can echo keys.
                        raise CompletionError(f'Model call failed: {type(exc).__name__}, status={status}') from None
                    time.sleep(self.retry_delay * 2 ** attempt)

    def close(self):
        self.tracer.flush()
        self.client.close()
