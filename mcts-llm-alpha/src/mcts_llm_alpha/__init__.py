"""MCTS-LLM Alpha挖掘主包（论文协议实现）。"""

__version__ = "0.2.0"

# Market and API dependencies are imported only when requested.
from importlib import import_module

_EXPORTS = {
    "load_config": ".config", "Config": ".config",
    "LLMClient": ".llm", "MCTSSearch": ".mcts", "MCTSNode": ".mcts", "SearchSettings": ".mcts",
    "FrequentSubtreeMiner": ".mcts", "ExperimentLedger": ".mcts",
    "QlibMarket": ".evaluation", "FactorRepository": ".evaluation", "EffectiveCriteria": ".evaluation",
    "run_experiment": ".experiment",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


__all__ = ["__version__", *_EXPORTS]
