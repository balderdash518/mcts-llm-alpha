"""Offline checks of the paper protocol and of what the LLM actually receives.

A scripted agent replaces the model and a synthetic panel replaces Qlib, so these
tests verify information flow and search invariants, not profitability.
"""
import pickle
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mcts_llm_alpha.evaluation.protocol import (EffectiveCriteria, EvaluationError, FactorRepository,
                                                RepositoryAlpha, daily_turnover, measure)
from mcts_llm_alpha.formula import check, structure, substitute, to_qlib, validate
from mcts_llm_alpha.llm.brief import render_brief
from mcts_llm_alpha.llm.client import GenerationError
from mcts_llm_alpha.mcts import ExperimentLedger, FrequentSubtreeMiner, MCTSSearch, SearchSettings
from mcts_llm_alpha.mcts.fsa import extract_root_genes

FIELDS = ("open", "high", "low", "close", "volume")


class FakeMarket:
    """Synthetic panel: formulas containing ``Ma`` carry signal, ``Log`` fails to compute."""

    def __init__(self, days=160, stocks=60):
        rng = np.random.default_rng(7)
        index = pd.date_range("2015-01-01", periods=days, freq="B")
        self.label_frame = pd.DataFrame(rng.normal(size=(days, stocks)), index=index,
                                        columns=[f"S{i}" for i in range(stocks)])
        self.fields, self.horizon, self.universe = FIELDS, 10, "synthetic"
        self.periods = {"train": ("2015-01-01", "2015-08-01"), "test": ("2015-08-02", "2015-12-31")}
        self.calls = []

    def measure(self, formula, period="train", repository=None):
        self.calls.append((formula, period, len(repository) if repository is not None else None))
        if "Log(" in formula:
            raise EvaluationError("Qlib 计算失败：log of non-positive values")
        seed = sum(map(ord, formula)) % (2 ** 32)
        noise = np.random.default_rng(seed).normal(size=self.label_frame.shape)
        strength = 0.6 if "Ma(" in formula else 0.0
        factor = self.label_frame * strength + pd.DataFrame(noise, index=self.label_frame.index,
                                                            columns=self.label_frame.columns)
        return measure(formula, factor, self.label_frame, repository, min_days=20)


class ScriptedAgent:
    """Stands in for LLMClient; formulas are consumed in order."""

    def __init__(self, formulas, corrections=(), fail_suggestions=False):
        self.formulas, self.corrections = list(formulas), list(corrections)
        self.fail_suggestions = fail_suggestions
        self.briefs, self.correction_errors, self.reviews, self.examples = [], [], [], []
        self.summaries = 0
        self.tracer = None

    def initial_portrait(self, fields, avoid, target="", explored=()):
        self.root_requests = getattr(self, "root_requests", []) + [(target, list(explored))]
        return {"name": "root", "description": "根假设", "pseudo_code": ["a = Ma(input=[close])"]}

    def _next(self):
        formula = self.formulas.pop(0)
        return {"formula": formula, "parameters": {"w": {}}, "candidates": [{"w": 5}, {"w": 10}, {"w": 20}]}

    def formula(self, portrait, fields, avoid, notes=()):
        return self._next()

    def suggestions(self, build, dimension, examples, avoid, fields=None):
        if self.fail_suggestions:
            raise GenerationError("refinement-suggestions: 模型未返回可用的 JSON")
        self.briefs.append(build(0))
        self.examples.append((dimension, examples))
        return {"diagnosis": "测试", "suggestions": [f"改进{dimension}"],
                "expected_effect": {dimension: "+"}}

    def refined_portrait(self, formula, suggestions, fields, avoid):
        return {"name": "child", "description": f"由 {formula} 改进", "pseudo_code": ["x"]}

    def correct(self, portrait, formula, parameters, error, fields, avoid, tried=(), build_brief=None):
        self.correction_errors.append(error)
        return {"formula": self.corrections.pop(0), "parameters": {}, "candidates": [{"w": 7}]}

    def summary(self, parent, child, dimension, deltas, changes):
        self.summaries += 1
        return f"{parent.label}->{child.label} 调整了 {dimension}"

    def review_overfitting(self, formula, history, neighbors):
        self.reviews.append((formula, list(history), list(neighbors)))
        return 6.0, "结构清晰"


def search_with(agent, tmp_path=None, **settings):
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl" if tmp_path else None, {"test": True})
    defaults = dict(total_search_count=4, random_seed=1)
    defaults.update(settings)
    return MCTSSearch(agent, FakeMarket(), ledger, settings=SearchSettings(**defaults), fields=FIELDS)


# ----------------------------------------------------------------- formula language
@pytest.mark.parametrize("formula, message", [
    ("Delay($close, -1)", "未来数据"), ("Foo($close, 5)", "未知算子"), ("Ma($vwap, 5)", "不可用"),
    ("Ma($close, a) + Std($open, b) + Sum($low, c) + Max($high, d)", "参数数量"),
    ("$close > $open", "比较运算符"), ("Ma($close, $open)", "窗口参数"), ("5 + 3", "数据字段"),
])
def test_invalid_formulas_return_actionable_messages(formula, message):
    ok, error = validate(formula, FIELDS)
    assert not ok and message in error


def test_paper_operators_translate_to_qlib_and_share_structure():
    assert to_qlib("Pct($close, 5)") == "($close / Ref($close, 5) - 1)"
    assert to_qlib("Autocorr($volume, 20, 1)") == "Corr($volume, Ref($volume, 1), 20)"
    assert to_qlib("-Greater($close, Ma($open, 3))") == "(0 - Gt($close, Mean($open, 3)))"
    assert to_qlib("Delay($close, 2) ** 2") == "Power(Ref($close, 2), 2)"
    assert substitute("Ma($close, w1) / Std($close, w2)", {"w1": 5, "w2": 20}) == \
        "(Ma($close, 5) / Std($close, 20))"
    assert structure("Ma($close, w1)") == structure("Ma($close, 30)") == "Ma($close, t)"
    assert check("Ma($close, w) * k", FIELDS) == {"w", "k"}


# --------------------------------------------------------------------------- metrics
def test_turnover_is_holdings_change_and_relative_rank_matches_eq3():
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    stable = pd.DataFrame(np.tile(np.arange(10.0), (30, 1)), index=index)
    flipping = pd.DataFrame([np.arange(10.0) * (1 if i % 2 else -1) for i in range(30)], index=index)
    assert daily_turnover(stable) == pytest.approx(0.0)
    assert daily_turnover(flipping) == pytest.approx(2.0)
    repo = FactorRepository()
    assert repo.scores({"rank_ic": 0.0, "rank_ir": 0.0, "turnover": 2.0, "max_corr": 1.0}) == \
        {"Effectiveness": 10.0, "Stability": 10.0, "Turnover": 10.0, "Diversity": 10.0}
    ranked = np.zeros((2, 2), dtype=np.float32)
    for i, ic in enumerate([0.01, 0.02, 0.02, 0.05]):
        repo.alphas.append(RepositoryAlpha(str(i), f"f{i}", {"rank_ic": ic, "rank_ir": 0.4,
                                                          "turnover": 0.5}, ranked))
        repo._corr[str(i)] = {}
    # Ties are not "better": 1 - share of strictly better alphas (one of four).
    assert repo.relative_rank("rank_ic", 0.02) == pytest.approx(0.25)
    assert repo.scores({"rank_ic": 0.02, "rank_ir": 0.4, "turnover": 0.6, "max_corr": 0.0})["Turnover"] == 0.0


def test_relative_rank_criterion_waits_until_it_can_resolve_the_bound():
    ranked = np.zeros((2, 2), dtype=np.float32)
    repo = FactorRepository()
    repo.add(RepositoryAlpha("a", "F_a", {"rank_ic": 0.05, "rank_ir": 0.5, "turnover": 0.3}, ranked), {})
    weaker = {"rank_ic": 0.03, "rank_ir": 0.4, "turnover": 0.3, "max_corr": 0.1}
    assert EffectiveCriteria().failures(weaker, repo) == []
    literal = EffectiveCriteria(relative_rank_min_size=1).failures(weaker, repo)
    assert len(literal) == 2 and all(r.startswith("R_") for r in literal)


def test_effective_check_reports_every_failed_criterion():
    repo = FactorRepository()
    reasons = EffectiveCriteria().failures(
        {"rank_ic": 0.01, "rank_ir": 0.2, "turnover": 1.7, "max_corr": 0.85}, repo)
    assert len(reasons) == 4 and any("RankIC" in r for r in reasons) and any("换手" in r for r in reasons)
    assert EffectiveCriteria().failures(
        {"rank_ic": 0.02, "rank_ir": 0.5, "turnover": 0.4, "max_corr": 0.1}, repo) == []


# ------------------------------------------------------------------------------ FSA
def test_fsa_abstracts_symbolic_and_numeric_parameters_and_keeps_closed_genes():
    assert extract_root_genes("Ma($close, w1)") == extract_root_genes("Ma($close, 20)")
    miner = FrequentSubtreeMiner(top_k=3, min_support=2).fit([
        "Std(Ma($close, 5), 10) / $volume", "Std(Ma($close, 20), 30) + $open", "Ma($close, 3) * $high"])
    avoid = miner.avoid_patterns()
    assert avoid[0] == "Ma($close, t)"
    # Std(Ma($close,t),t) has support 2 and does not hide Ma (support 3), so both are closed.
    assert "Std(Ma($close, t), t)" in avoid


# --------------------------------------------------------------------------- ledger
def test_ledger_folds_structures_groups_failures_and_reloads_by_scope(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger(path, {"protocol": 1})
    for window, reward in ((5, 4.0), (10, 6.0)):
        a = ledger.propose(source="refinement", formula=f"Ma($close, {window})",
                           structure="Ma($close, t)", root_genes=["Ma($close, t)"])
        ledger.record(a, "evaluated", reward=reward, metrics={"rank_ic": 0.02})
        ledger.record(a, "rejected", reasons=["RankIR 0.1 < 0.3"])
    for value in (3, 4):
        ledger.failed(ledger.propose(source="refinement", formula=f"Log($close - {value})"),
                      f"覆盖率只有 {value}%", "evaluation")
    ledger.propose(source="parameter_trial", formula="Ma($close, 99)", structure="Ma($close, t)")
    found = ExperimentLedger(path, {"protocol": 1}).retrieve(["Ma($close, t)"])
    assert [(r["formula"], r["count"]) for r in found["rejected"]] == [("Ma($close, 10)", 2)]
    assert len(found["execution_failed"]) == 1 and found["execution_failed"][0]["count"] == 2
    assert ExperimentLedger(path, {"protocol": 2}).records == {}


# ----------------------------------------------------------------- search behaviour
def test_multi_tree_search_admits_after_each_tree_and_counts_generations(tmp_path):
    agent = ScriptedAgent(["Ma($close, w)", "Std($open, w)", "Ma($high, w) * $volume",
                           "Ma($low, w)", "Sum($close, w)", "Ma($open, w) - $close", "Std($low, w)",
                           "Ma($volume, w)"] * 3)
    search = search_with(agent, tmp_path, total_search_count=8, seed_formula="Ma($close, 5)")
    repository = search.run()
    assert search.search_count == 8 and search.tree_index >= 2
    assert search.trees[0].formula == "Ma($close, 5)"          # user seed roots tree 0
    sizes = {call[2] for call in search.market.calls if call[0] in
             {n.formula for n in search.trees[0].walk()}}
    assert sizes == {0}                                          # tree 0 scored on an empty snapshot
    assert len(repository) > 0 and all("Ma(" in a.formula for a in repository)
    for root in search.trees:
        for node in root.walk():
            assert node.admission and node.ranked is None and node.score is not None
    statuses = {r.get("status") for r in search.ledger.records.values() if r.get("source") == "tree_node"}
    assert statuses <= {"admitted", "rejected"}


def test_empty_repository_uses_measured_candidates_as_reference(tmp_path):
    agent = ScriptedAgent([f"Std($open, w) + {i}" if i % 2 else f"Sum($low, w) * {i}" for i in range(1, 30)])
    search = search_with(agent, tmp_path, total_search_count=10)
    first = search.run_tree()
    assert len(search.repository) == 0
    # The very first tree ranks against its root's three parameter trials.
    root_trials = [r["metrics"]["rank_ic"] for r in search.ledger.records.values()
                   if r.get("source") in ("parameter_trial", "tree_node") and r.get("label") in (None, "V0")
                   and r.get("tree") == 0 and r.get("parent") is None and r.get("metrics")]
    assert len(root_trials) == 3
    for node in first.walk():
        better = sum(v > node.metrics["rank_ic"] for v in root_trials) / 3
        assert node.scores["Effectiveness"] == pytest.approx(10 * (1 - better))
    second = search.run_tree()
    measured = [r["metrics"]["rank_ic"] for r in search.ledger.records.values()
                if r.get("metrics") and r.get("tree") == 0 and r.get("status") != "execution_failed"]
    assert len(search.reference) == len(measured)
    for node in second.walk():
        better = sum(v > node.metrics["rank_ic"] for v in measured) / len(measured)
        assert node.scores["Effectiveness"] == pytest.approx(10 * (1 - better))


def test_parameter_ties_are_broken_by_raw_rank_ir(tmp_path):
    agent = ScriptedAgent(["Std($open, w)"])
    search = search_with(agent, tmp_path, total_search_count=1)
    info = {"formula": "Std($open, w)", "parameters": {}, "candidates": [{"w": 5}, {"w": 10}, {"w": 20}]}
    chosen = search.try_parameters(info, 0, None, None, {"description": ""})
    trials = [search.market.measure(f"Std($open, {w})") for w in (5, 10, 20)]
    assert chosen["measurement"].metrics["rank_ir"] == max(t.metrics["rank_ir"] for t in trials)


def test_later_roots_see_earlier_roots_and_the_prediction_target(tmp_path):
    agent = ScriptedAgent(["Std($open, w)", "Sum($low, w) * 2", "Std($high, w) + 3"] * 6)
    search = search_with(agent, tmp_path, total_search_count=8)
    search.run_tree()
    search.run_tree()
    first, second = agent.root_requests[:2]
    assert "未来 10 个交易日收益" in first[0] and first[1] == []
    assert len(second[1]) == 1 and second[1][0].startswith("Std($open, ")


def test_budget_grows_on_new_tree_best_and_backprop_keeps_max(tmp_path):
    agent = ScriptedAgent(["Std($open, w)", "Ma($close, w)", "Ma($high, w)", "Ma($low, w)",
                           "Ma($open, w)", "Ma($volume, w)"] + ["Std($low, w)"] * 30)
    search = search_with(agent, tmp_path, total_search_count=6)
    root = search.run_tree()
    nodes = list(root.walk())
    assert len(nodes) > 3                                        # B=3 was extended by b=1
    assert root.q == pytest.approx(max(n.score for n in nodes))
    assert root.visits == len(nodes)


def test_second_expansion_sees_first_child_lineage_and_no_identifiers(tmp_path):
    agent = ScriptedAgent(["Std($open, w)"] * 10)
    search = search_with(agent, tmp_path, total_search_count=4)
    search.run_tree()
    later = agent.briefs[-1]
    assert "【评估协议】" in later and "未来 10 个交易日收益" in later
    assert "V0→V1" in later       # as lineage of V1, or as a child already tried from V0
    assert not re.search(r"[0-9a-f]{32}", later)                # no attempt ids in prompts
    for brief in agent.briefs:
        for edge in set(re.findall(r"V\d+→V\d+", brief)):
            assert brief.count(edge + " [") == 1                 # each edge rendered once


def test_invalid_formula_error_reaches_correction_and_failures_are_not_defaults(tmp_path):
    agent = ScriptedAgent(["Ma($close, w)", "Log($close - $open)"], corrections=["Ma($open, w) * $close"])
    search = search_with(agent, tmp_path, total_search_count=2, tree_initial_budget=1)
    search.run_tree()
    assert agent.correction_errors and "log of non-positive" in agent.correction_errors[0]
    child = search.trees[0].children[0]
    assert child.formula == "(Ma($open, 7) * $close)"
    failed = [r for r in search.ledger.records.values() if r.get("status") == "execution_failed"]
    assert failed and all(r.get("scores") is None for r in failed)


def test_generation_failure_creates_no_node(tmp_path):
    agent = ScriptedAgent(["Ma($close, w)"], fail_suggestions=True)
    search = search_with(agent, tmp_path, total_search_count=3)
    root = search.run_tree()
    assert root.children == [] and search.search_count == 1
    assert any(r.get("stage") == "suggestion" for r in search.ledger.records.values())


def test_overfitting_review_sees_history_and_neighbours(tmp_path):
    agent = ScriptedAgent(["Std($open, w)"] * 10)
    search = search_with(agent, tmp_path, total_search_count=5)
    search.run_tree()
    formula, history, neighbors = agent.reviews[-1]
    assert history and history != ["无改进历史（初始公式）"]
    assert agent.reviews[0][1] == ["无改进历史（初始公式）"]
    assert any(n for _, _, n in agent.reviews)


def test_few_shot_follows_appendix_c2(tmp_path):
    agent = ScriptedAgent(["Std($open, w)"] * 10)
    search = search_with(agent, tmp_path)
    node = SimpleNamespace(correlations={"a": 0.9, "b": 0.1, "c": 0.5})
    ranked = np.zeros((2, 2), dtype=np.float32)
    for name, ic in (("a", 0.09), ("b", 0.01), ("c", 0.03)):
        search.repository.add(RepositoryAlpha(name, f"F_{name}", {"rank_ic": ic, "rank_ir": ic * 10,
                                                                "turnover": 0.3}, ranked), {})
    assert search.few_shot(node, "Turnover") == [] and search.few_shot(node, "Overfitting") == []
    assert search.few_shot(node, "Diversity")[0]["formula"] == "F_b"
    assert search.few_shot(node, "Effectiveness")[0]["formula"] == "F_c"   # top-50% corr removed


def test_brief_levels_shrink_lower_priority_sections(tmp_path):
    agent = ScriptedAgent(["Std($open, w)"] * 12)
    search = search_with(agent, tmp_path, total_search_count=8, max_tree_iterations=8)
    root = search.run_tree()
    deepest = max(root.walk(), key=lambda n: len(n.lineage()))
    full = render_brief(deepest, search.protocol, 0, level=0)
    tight = render_brief(deepest, search.protocol, 0, level=3)
    assert len(tight) < len(full) and "【当前因子" in tight and "【评估协议】" in tight


def test_checkpoint_round_trip(tmp_path):
    agent = ScriptedAgent(["Ma($close, w)", "Ma($open, w)", "Ma($high, w)", "Std($low, w)"] * 4)
    search = search_with(agent, tmp_path, total_search_count=4,
                         checkpoint_path=str(tmp_path / "checkpoint.pkl"))
    search.run()
    resumed = search_with(ScriptedAgent([]), tmp_path, total_search_count=4)
    resumed.load_checkpoint(tmp_path / "checkpoint.pkl")
    assert resumed.search_count == search.search_count and len(resumed.repository) == len(search.repository)
    assert resumed.avoid == search.avoid
    assert pickle.loads(pickle.dumps(resumed.repository)).snapshot_ids() == search.repository.snapshot_ids()


def test_intent_is_judged_on_raw_metrics_when_scores_do_not_move():
    from mcts_llm_alpha.llm.brief import fmt_expectation
    from mcts_llm_alpha.mcts.node import MCTSNode
    parent = MCTSNode("Ma($close, 5)")
    child = MCTSNode("Ma($close, 20)", parent, "Turnover")
    for node, turnover in ((parent, 0.8), (child, 0.7)):
        node.metrics = {"rank_ic": 0.01, "rank_ir": 0.1, "turnover": turnover, "max_corr": 0.0}
        node.scores = {"Turnover": 3.3, "Overfitting": 6.0}
    child.suggestions = {"expected_effect": {"Turnover": "+", "Overfitting": "+"}}
    text = fmt_expectation(child)
    assert "日换手率 -0.1000达成" in text and "分数 +0.0未达成" in text
