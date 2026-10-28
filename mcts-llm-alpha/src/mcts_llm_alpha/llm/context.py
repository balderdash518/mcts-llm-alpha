"""Prompt budgeting. Evidence is shortened by re-rendering the SearchBrief at a
coarser level (see ``brief.LEVELS``), never by cutting text mid-record."""
import math


def estimate_tokens(text):
    # Approximation, not the provider tokenizer. CJK needs more tokens than English.
    ascii_count = sum(ord(c) < 128 for c in text)
    return math.ceil(ascii_count / 2.5 + (len(text) - ascii_count) * 1.5) + 64


def prompt_budget(context_window, min_output_tokens, safety_tokens=256):
    return context_window - min_output_tokens - safety_tokens


def fit_prompt(prompt, context_window, min_output_tokens, safety_tokens=256):
    if estimate_tokens(prompt) > prompt_budget(context_window, min_output_tokens, safety_tokens):
        raise ValueError('Prompt exceeds model context; reduce input or min_output_tokens')
    return prompt, 0


def fit_levels(build, context_window, min_output_tokens, levels=4):
    """Return the most detailed rendering ``build(level)`` that fits the context."""
    budget = prompt_budget(context_window, min_output_tokens)
    for level in range(levels):
        prompt = build(level)
        if estimate_tokens(prompt) <= budget:
            return prompt, level
    raise ValueError('Mandatory evidence exceeds model context even at the coarsest level')
