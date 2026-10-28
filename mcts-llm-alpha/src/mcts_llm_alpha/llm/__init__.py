"""LLM 角色：画像、改进建议、改进、公式、纠错、改进总结与过拟合审阅。"""

from .client import LLMClient, GenerationError

__all__ = ["LLMClient", "GenerationError"]
