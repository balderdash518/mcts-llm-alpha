"""LLM roles used by the search: portrait, suggestions, refinement, formula,
correction, refinement summary and overfitting review.

Every structured answer is parsed strictly. A response that cannot be parsed is
retried and then raised as GenerationError; nothing is replaced by a default
formula or score, so the search never records an action the model did not take.
"""

import json
import re

from .transport import CompletionTransport, CompletionError
from .context import fit_levels
from . import prompts


class GenerationError(RuntimeError):
    """The model did not return a usable structured answer."""


def parse_json(text):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("回答中没有 JSON 对象")
    return json.loads(text[start:end + 1])


DEFAULT_TEMPERATURES = {
    # Appendix F.1: portraits/formulas 1.0, correction 0.8, overfitting review 0.1.
    "portrait": 1.0, "formula": 1.0, "suggestion": 1.0, "refinement": 1.0,
    "correction": 0.8, "review": 0.1, "summary": 0.7,
}


class LLMClient:
    def __init__(self, api_key, model="gpt-4.1", max_retries=3, retry_delay=1.0,
                 temperatures=None, json_retries=2, extra_body_by_role=None, **options):
        self.transport = CompletionTransport(api_key, model, max_retries, retry_delay, **options)
        self.client = self.transport.client
        self.model = model
        self.tracer = self.transport.tracer
        self.temperatures = {**DEFAULT_TEMPERATURES, **(temperatures or {})}
        self.json_retries = json_retries
        self.extra_body_by_role = extra_body_by_role or {}

    @classmethod
    def from_config(cls, config):
        import os
        key = os.getenv(config.api_key_env)
        if not key or key.startswith("your_"):
            raise ValueError(f"Configure {config.api_key_env} in the project .env")
        return cls(api_key=key, model=config.model, max_retries=config.max_retries,
                   retry_delay=config.retry_delay, temperatures=config.temperature,
                   base_url=config.base_url, timeout_s=config.timeout_s, stream=config.stream,
                   max_tokens=config.max_tokens, context_window=config.context_window,
                   min_output_tokens=config.min_output_tokens, extra_body=config.extra_body,
                   extra_body_by_role=config.extra_body_by_role,
                   reasoning_effort=config.reasoning_effort, langfuse=config.langfuse)

    def close(self):
        self.transport.close()

    def get_completion_sync(self, prompt, temperature=0.7, max_tokens=None, name="model-completion",
                            role=None):
        extra = getattr(self, "extra_body_by_role", {}).get(role)
        if extra:
            return self.transport.complete(prompt, temperature, max_tokens, name, extra_body=extra)
        return self.transport.complete(prompt, temperature, max_tokens, name)

    def fit(self, build):
        """Render the most detailed prompt variant that fits the context window."""
        return fit_levels(build, self.transport.context_window, self.transport.min_output_tokens)[0]

    def _json(self, prompt, role, name, required, validate=None):
        last = None
        for _ in range(self.json_retries + 1):
            try:
                text = self.get_completion_sync(prompt, self.temperatures[role], name=name, role=role)
            except CompletionError as error:
                # Truncated (length-limited) or failed calls end this attempt, not the run;
                # a length-limited answer is not retried because it would repeat the cost.
                raise GenerationError(f"{name}: {error}") from None
            try:
                data = parse_json(text)
                missing = [k for k in required if k not in data]
                if missing:
                    raise ValueError(f"缺少字段 {missing}")
                if validate:
                    validate(data)
                return data
            except (ValueError, TypeError, KeyError) as error:
                last = error
        raise GenerationError(f"{name}: 模型未返回可用的 JSON（{last}）")

    @staticmethod
    def _check_portrait(data):
        if not isinstance(data["pseudo_code"], list) or not data["pseudo_code"]:
            raise ValueError("pseudo_code 必须是非空列表")

    @staticmethod
    def _check_formula(data):
        if not isinstance(data["formula"], str) or not data["formula"].strip():
            raise ValueError("formula 为空")
        candidates = data["candidates"]
        if not isinstance(candidates, list) or not candidates or \
                not all(isinstance(c, dict) for c in candidates):
            raise ValueError("candidates 必须是参数字典列表")

    def initial_portrait(self, fields, avoid, target="", explored=()):
        return self._json(prompts.initial_portrait_prompt(fields, avoid, target, explored), "portrait",
                          "initial-portrait", ("name", "description", "pseudo_code"),
                          self._check_portrait)

    def formula(self, portrait, fields, avoid, notes=()):
        return self._json(prompts.formula_prompt(portrait, fields, avoid, notes), "formula",
                          "portrait-to-formula", ("formula", "parameters", "candidates"),
                          self._check_formula)

    def suggestions(self, build_brief, dimension, examples, avoid, fields):
        prompt = self.fit(lambda level: prompts.suggestion_prompt(
            build_brief(level), dimension, examples, avoid, fields))

        def check(data):
            if not isinstance(data["suggestions"], list) or not data["suggestions"]:
                raise ValueError("suggestions 必须是非空列表")
            if not isinstance(data.get("expected_effect", {}), dict):
                raise ValueError("expected_effect 必须是对象")
        return self._json(prompt, "suggestion", "refinement-suggestions",
                          ("diagnosis", "suggestions"), check)

    def refined_portrait(self, formula, suggestions, fields, avoid):
        return self._json(prompts.refinement_prompt(formula, suggestions, fields, avoid),
                          "refinement", "refine-portrait", ("name", "description", "pseudo_code"),
                          self._check_portrait)

    def correct(self, portrait, formula, parameters, error, fields, avoid, tried=(), build_brief=None):
        prompt = self.fit(lambda level: prompts.correction_prompt(
            portrait, formula, parameters, error, fields, avoid, tried,
            build_brief(level + 1) if build_brief else ""))
        return self._json(prompt, "correction", "correct-formula",
                          ("formula", "parameters", "candidates"), self._check_formula)

    def summary(self, parent, child, dimension, deltas, score_changes):
        try:
            text = self.get_completion_sync(
                prompts.summary_prompt(parent, child, dimension, deltas, score_changes),
                self.temperatures["summary"], name="refinement-summary", role="summary").strip()
        except CompletionError as error:
            raise GenerationError(f"refinement-summary: {error}") from None
        if not text:
            raise GenerationError("refinement-summary: 模型返回为空")
        return text

    def review_overfitting(self, formula, history, neighbors):
        def check(data):
            score = float(data["score"])
            if not 0 <= score <= 10:
                raise ValueError("score 必须在 0-10 之间")
        data = self._json(prompts.overfitting_prompt(formula, history, neighbors), "review",
                          "overfitting-review", ("score", "reason"), check)
        return float(data["score"]), str(data["reason"])
