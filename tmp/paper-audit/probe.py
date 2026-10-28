"""Read-only audit probes; use synthetic data and never call a remote LLM."""
import ast
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mcts-llm-alpha/src/mcts_llm_alpha"
results = {}

# Bypass the broken top-level __init__ only for isolated component inspection.
for name, rel in [("mcts_llm_alpha", ""),
                  ("mcts_llm_alpha.evaluation", "evaluation"),
                  ("mcts_llm_alpha.llm", "llm"),
                  ("mcts_llm_alpha.config", "config")]:
    package = types.ModuleType(name)
    package.__path__ = [str(SRC / rel)]
    sys.modules[name] = package

class ConfigFixture:
    def __init__(self):
        self.evaluation = types.SimpleNamespace(effectiveness_threshold=3.)
sys.modules["mcts_llm_alpha.config"].Config = ConfigFixture

rank_mod = importlib.import_module("mcts_llm_alpha.evaluation.relative_ranking")
turn_mod = importlib.import_module("mcts_llm_alpha.evaluation.metrics.turnover")
cache_mod = importlib.import_module("mcts_llm_alpha.evaluation.cache")
config_mod = importlib.import_module("mcts_llm_alpha.config")
comp_mod = importlib.import_module("mcts_llm_alpha.evaluation.comprehensive")
search_mod = importlib.import_module("mcts_llm_alpha.mcts.search")
node_mod = importlib.import_module("mcts_llm_alpha.mcts.node")
fsa_mod = importlib.import_module("mcts_llm_alpha.mcts.fsa")
validator = importlib.import_module("mcts_llm_alpha.formula.validator")

dates = pd.date_range("2020-01-01", periods=40)
stocks = ["A", "B", "C"]
idx = pd.MultiIndex.from_product([stocks, dates], names=["instrument", "datetime"])
stable = pd.DataFrame({"factor": np.repeat([1., 2., 3.], len(dates))}, index=idx)
flip_values = []
for stock_i in range(3):
    flip_values.extend(stock_i + 1 if d % 2 == 0 else 3 - stock_i for d in range(len(dates)))
flipping = pd.DataFrame({"factor": flip_values}, index=idx)
ranking = rank_mod.RelativeRankingEvaluator()
stable_metric = turn_mod.calc_turnover(stable)
flip_metric = turn_mod.calc_turnover(flipping)
results["turnover_direction"] = {
    "stable_metric": stable_metric,
    "flipping_metric": flip_metric,
    "stable_score": ranking.calculate_relative_rank(stable_metric, "Turnover") * 10,
    "flipping_score": ranking.calculate_relative_rank(flip_metric, "Turnover") * 10,
}
assert results["turnover_direction"]["stable_score"] < results["turnover_direction"]["flipping_score"]

ranking.effective_repository = [{"Diversity": 1., "IC": .03}] * 5
results["ties"] = {
    "diversity_score_all_equal": ranking.calculate_relative_rank(1., "Diversity") * 10,
    "paper_1_minus_strict_better_rank": 10.,
}
assert results["ties"]["diversity_score_all_equal"] == 0

with contextlib.redirect_stdout(io.StringIO()):
    evaluator = comp_mod.ComprehensiveEvaluator(config_mod.Config())
    evaluator._evaluate_with_qlib = lambda formula, repo: (
        {"IC": .04, "IR": .6, "Turnover": .6, "Diversity": 1., "Overfitting": .9}, stable)
    scores_by_repeat = []
    for _ in range(6):
        scores_by_repeat.append(evaluator.evaluate_formula("same_formula", [], None)[0])
results["evaluation_mutates_ranking_repository"] = {
    "evaluations_of_identical_formula": 6,
    "effective_repository_entries": len(evaluator.ranking_evaluator.effective_repository),
    "effectiveness_sequence": [s["Effectiveness"] for s in scores_by_repeat],
    "diversity_sequence": [s["Diversity"] for s in scores_by_repeat],
}
assert scores_by_repeat[-1]["Effectiveness"] == 0

qlib_mod = importlib.import_module("mcts_llm_alpha.evaluation.qlib_evaluator")
class FakeD:
    calls = 0
    @staticmethod
    def features(universe, fields, **kwargs):
        FakeD.calls += 1
        if fields == ["Ref($close, -1) / $close - 1"]:
            return pd.DataFrame({"return": np.random.default_rng(42).normal(size=len(idx))}, index=idx)
        return stable.copy()
qlib_mod.D = FakeD
qlib_mod._global_cache = cache_mod.FormulaEvaluationCache()
# Repo-sensitive fixture: empty zoo is fully diverse; an exact copy is redundant.
# This isolates cache correctness without needing Qlib or SciPy installed.
qlib_mod.calc_diversity = lambda factor, repo: 0. if repo else 1.
with contextlib.redirect_stdout(io.StringIO()):
    a = qlib_mod.evaluate_formula_qlib("$close", [], "2020-01-01", "2020-01-25", stocks, ic_method="pearson")
    b = qlib_mod.evaluate_formula_qlib("$close", [stable], "2020-01-01", "2020-01-25", stocks, ic_method="pearson")
    fresh = qlib_mod.evaluate_formula_qlib("$close", [stable], "2020-01-01", "2020-01-25", stocks, ic_method="pearson", use_cache=False)
results["cache_repo_dependency"] = {
    "empty_repo_diversity": a[0]["Diversity"],
    "identical_factor_in_repo_cached_diversity": b[0]["Diversity"],
    "identical_factor_in_repo_fresh_diversity": fresh[0]["Diversity"],
}
assert b[0]["Diversity"] == 1 and fresh[0]["Diversity"] < 1e-6

# Wrapper accepts a stub type instead of importing the unavailable OpenAI SDK.
client_stub = types.ModuleType("mcts_llm_alpha.llm.client")
client_stub.LLMClient = object
sys.modules[client_stub.__name__] = client_stub
wrapper = importlib.import_module("mcts_llm_alpha.llm.wrapper")
class FakeClient:
    def refine_formula(self, *args, **kwargs):
        return "Mean($volume,w1)", "portrait", "description", {
            "symbolic_formula": "Mean($volume,w1)", "selected_params": {"w1": 30}}
parent = node_mod.MCTSNode("Mean($close,w1)")
parent.formula_info = {"symbolic_formula": parent.formula, "selected_params": {"w1": 5}}
before = dict(parent.formula_info)
wrapper.create_formula_refiner(FakeClient())(parent, "Diversity")
results["parent_formula_info_mutation"] = {
    "parent_formula": parent.formula,
    "before": before,
    "after": parent.formula_info,
}
assert parent.formula_info["symbolic_formula"] != parent.formula

source = ast.parse((SRC / "mcts/search.py").read_text(encoding="utf-8"))
cls = next(n for n in source.body if isinstance(n, ast.ClassDef))
duplicates = [n.lineno for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate_valid_formula"]
results["shadowed_validator"] = {"definitions": duplicates,
    "active_definition_has_validation": "validate_formula" in search_mod.MCTSSearch.generate_valid_formula.__code__.co_names}
assert len(duplicates) == 2 and not results["shadowed_validator"]["active_definition_has_validation"]
results["lookahead_validator"] = {s: validator.validate_formula(s)[0]
    for s in ["Ref($close, -1)", "Mean($close, 0)", "UnknownOp($close, 5)"]}

scores = dict.fromkeys(node_mod.MCTSNode.DIMENSIONS, 8.)
parent = node_mod.MCTSNode("$close")
parent.scores = scores
parent.factor = stable
child = parent.expand("Effectiveness", "$volume", scores, stable, "portrait", "change")
results["child_factor_alias"] = {"parent_has_factor": parent.factor is not None,
    "child_has_factor_returns": child.factor_returns is not None,
    "child_has_factor": child.factor is not None}
assert child.factor is None

seen = {}
def capture_evaluator(formula, repo, node):
    seen["history_present_at_evaluation"] = hasattr(node, "refinement_history")
    return scores.copy(), stable, {"IC": .04}
search = search_mod.MCTSSearch(lambda: None,
    lambda *args: ("Mean($volume,w1)", "portrait", "description", {"selected_params": {"w1": 5}}),
    capture_evaluator)
parent = node_mod.MCTSNode("$close")
parent.scores = scores.copy()
parent.refinement_history = [{"dimension": "Diversity", "description": "prior change"}]
with contextlib.redirect_stdout(io.StringIO()):
    child = search.expand_node(parent)
results["new_node_history_at_evaluation"] = seen
assert not seen["history_present_at_evaluation"]

tree_a, _ = fsa_mod.parse_formula_to_ast("Mean($close,w1)")
tree_b, _ = fsa_mod.parse_formula_to_ast("Mean($close,w2)")
results["symbolic_fsa_parameters"] = {
    "same_structure_w1": fsa_mod.extract_root_genes(tree_a),
    "same_structure_w2": fsa_mod.extract_root_genes(tree_b),
}
assert set(results["symbolic_fsa_parameters"]["same_structure_w1"]) != set(results["symbolic_fsa_parameters"]["same_structure_w2"])

out = ROOT / "tmp/paper-audit/probe-results.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(results, ensure_ascii=False, indent=2))
