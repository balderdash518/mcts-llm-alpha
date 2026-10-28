"""LLM-guided MCTS for alpha mining (Algorithm 1, Appendices B, C and F.1).

Mining runs many small search trees. Each tree starts from an LLM-generated root
(or the user seed for the first tree), has an initial budget B=3 that grows by
b=1 whenever a node beats the tree's best score, and is scored against a
repository snapshot that stays fixed while the tree is searched. When the tree
is finished every node is checked with the absolute effective-alpha criteria and
admitted to the repository in creation order; FSA patterns are then re-mined
from the repository for the next tree. Mining stops at the total search count
(number of alpha generations, i.e. tree nodes).
"""

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .node import MCTSNode
from .fsa import FrequentSubtreeMiner, extract_root_genes
from ..evaluation.protocol import (DIMENSIONS, METRIC_OF, EvaluationError, RepositoryAlpha,
                                   FactorRepository, MetricPool, EffectiveCriteria, rank_correlation)
from ..formula import FormulaError, check, substitute, structure
from ..llm.brief import render_brief, history_lines, edge_line, fmt_deltas
from ..llm.client import GenerationError
from ..observability import SearchTracer


@dataclass
class SearchSettings:
    total_search_count: int = 1000
    tree_initial_budget: int = 3
    budget_increment: int = 1
    exploration_constant: float = 1.0
    dimension_temperature: float = 1.0
    max_score: float = 10.0
    max_tree_iterations: int = 30
    max_corrections: int = 3
    max_root_attempts: int = 3
    few_shot_k: int = 1
    correlation_filter_ratio: float = 0.5
    seed_formula: str = None
    random_seed: int = 0
    checkpoint_path: str = None


class InvalidCandidate(Exception):
    def __init__(self, message, formula="", parameters=None):
        super().__init__(message)
        self.formula, self.parameters = formula, parameters or {}


class MCTSSearch:
    def __init__(self, agent, market, ledger, *, settings=None, criteria=None, fields=None,
                 protocol=None, tracer=None):
        self.agent, self.market, self.ledger = agent, market, ledger
        self.settings = settings or SearchSettings()
        self.criteria = criteria or EffectiveCriteria()
        self.fields = tuple(fields or market.fields)
        self.protocol = protocol or {
            "horizon": market.horizon, "train": market.periods["train"],
            "universe": market.universe}
        self.protocol["criteria"] = asdict(self.criteria)
        self.tracer = tracer or getattr(agent, "tracer", None) or SearchTracer("off")
        self.repository = FactorRepository()
        self.reference = self.repository
        self.fsa = FrequentSubtreeMiner()
        self.avoid = []
        self.rng = np.random.default_rng(self.settings.random_seed)
        self.search_count = 0
        self.tree_index = 0
        self.trees = []

    # ------------------------------------------------------------------ driver
    def run(self):
        while self.search_count < self.settings.total_search_count:
            self.run_tree()
            self.save_checkpoint()
        return self.repository

    def run_tree(self):
        s = self.settings
        tree_id = self.tree_index
        self.tree_index += 1
        self.reference = self.ranking_reference()
        with self.tracer.observe("mcts-tree", as_type="agent", input={
                "tree": tree_id, "search_count": self.search_count,
                "repository_size": len(self.repository), "reference_size": len(self.reference),
                "avoid": self.avoid}) as span:
            nodes = []
            root = self.create_root(tree_id, nodes)
            if not len(self.reference):
                # Nothing measured before this tree: rank against the root's parameter
                # trials, frozen from here on, and rescore the root on that reference.
                self.reference = MetricPool([r.get("metrics") for r in self.ledger.records.values()
                                             if r.get("tree") == tree_id and r.get("metrics")])
                root.scores.update(self.reference.scores(root.metrics))
                root.score = float(np.mean([root.scores[d] for d in DIMENSIONS]))
                self.record_node(root)
            root.visits, root.q = 1, root.score
            self.report(root, "root")
            budget, best, iteration = s.tree_initial_budget, root.score, 0
            while (iteration < budget and iteration < s.max_tree_iterations
                   and self.search_count < s.total_search_count):
                iteration += 1
                with self.tracer.observe("mcts-iteration", as_type="agent", input={
                        "tree": tree_id, "iteration": iteration, "budget": budget}) as obs:
                    selected, path = self.select(root)
                    child = self.expand(selected, nodes)
                    if child is None:
                        obs.update(output={"status": "no_valid_child", "selected": selected.label})
                        continue
                    if child.score > best:          # Algorithm 1, lines 23-26
                        budget += s.budget_increment
                        best = child.score
                    for node in path + [child]:     # lines 34-37
                        node.visits += 1
                        node.q = max(node.q, child.score)
                    self.report(child, f"from {selected.label}, budget {budget}")
                    obs.update(output={"selected": selected.label, "child": child.label,
                                       "formula": child.formula, "score": child.score})
            self.admit(nodes)
            self.trees.append(root)
            span.update(output={"nodes": len(nodes), "repository_size": len(self.repository)})
        return root

    def ranking_reference(self):
        """Repository, or (cold start) every candidate measured before this tree."""
        if len(self.repository):
            return self.repository
        return MetricPool([r.get("metrics") for r in self.ledger.records.values()
                           if r.get("metrics") and r.get("status") != "execution_failed"])

    @staticmethod
    def report(node, note):
        m = node.metrics
        print(f"[{node.name}] {note} | {node.dimension or 'root'} | {node.formula} | "
              f"RankIC {m['rank_ic']:.4f} RankIR {m['rank_ir']:.3f} 换手 {m['turnover']:.3f} "
              f"最大相关 {m['max_corr']:.3f} | 总分 {node.score:.2f}", flush=True)

    # --------------------------------------------------------------- selection
    def select(self, root):
        c = self.settings.exploration_constant
        node, path = root, [root]
        while node.children:
            best = max(node.children, key=lambda child: child.uct(c))
            if node.virtual_uct(c) >= best.uct(c):
                break
            node = best
            path.append(node)
        return node, path

    def dimension_probabilities(self, node):
        scores = np.array([node.scores[d] for d in DIMENSIONS], dtype=float)
        logits = (self.settings.max_score - scores) / self.settings.dimension_temperature
        weights = np.exp(logits - logits.max())
        return dict(zip(DIMENSIONS, weights / weights.sum()))

    # --------------------------------------------------------------- expansion
    def few_shot(self, node, dimension):
        """Appendix C.2 exemplar selection from the effective repository."""
        k, alphas = self.settings.few_shot_k, list(self.repository)
        if not alphas or dimension in ("Turnover", "Overfitting") or k <= 0:
            return []
        corr = lambda a: node.correlations.get(a.attempt_id, 0.0)
        if dimension == "Diversity":
            chosen, reason = sorted(alphas, key=corr)[:k], "与当前因子相关性最低"
        else:
            ordered = sorted(alphas, key=corr, reverse=True)
            kept = ordered[int(len(ordered) * self.settings.correlation_filter_ratio):] or ordered
            metric = METRIC_OF[dimension]
            chosen = sorted(kept, key=lambda a: a.metrics[metric], reverse=True)[:k]
            reason = f"相关性较低的一半中 {dimension} 最好"
        return [{"reason": f"{reason}（相关 {corr(a):.2f}）", "formula": a.formula,
                 "metrics": a.metrics, "description": a.description} for a in chosen]

    def brief_builder(self, node, dimension=None, probabilities=None):
        retrieved = self.ledger.retrieve(node.root_genes, dimension=dimension)

        def build(level):
            return render_brief(node, self.protocol, len(self.repository), len(self.reference),
                                dimension=dimension,
                                probabilities=probabilities, retrieved=retrieved, avoid=self.avoid,
                                level=level)
        return build

    def expand(self, node, nodes):
        probabilities = self.dimension_probabilities(node)
        dimension = str(self.rng.choice(DIMENSIONS, p=list(probabilities.values())))
        build = self.brief_builder(node, dimension, probabilities)
        stage = "suggestion"
        attempt = self.ledger.propose(source="refinement", tree=node.tree, parent=node.label,
                                      parent_attempt=node.attempt_id, dimension=dimension)
        try:
            suggestions = self.agent.suggestions(build, dimension, self.few_shot(node, dimension),
                                                 self.avoid, self.fields)
            stage = "refinement"
            portrait = self.agent.refined_portrait(node.formula, suggestions, self.fields, self.avoid)
        except GenerationError as error:
            self.ledger.failed(attempt, error, stage)
            return None
        candidate = self.materialize(portrait, node.tree, dimension, node, build, attempt)
        if candidate is None:
            return None
        child = self.make_node(candidate, node.tree, len(nodes), parent=node, dimension=dimension)
        child.suggestions = suggestions
        nodes.append(child)
        if not self.review(child):
            node.children.remove(child)
            nodes.remove(child)
            return None
        self.search_count += 1
        self.ledger.record(attempt, "materialized", node_attempt=child.attempt_id)
        try:
            child.summary = self.agent.summary(
                node, child, dimension, fmt_deltas(child, node),
                {d: round(child.scores[d] - node.scores[d], 2) for d in DIMENSIONS})
        except GenerationError as error:
            self.ledger.record(child.attempt_id, "evaluated", summary_error=str(error))
        self.record_node(child)
        return child

    def create_root(self, tree_id, nodes):
        seed = self.settings.seed_formula if tree_id == 0 else None
        for _ in range(self.settings.max_root_attempts):
            attempt = self.ledger.propose(source="root", tree=tree_id)
            try:
                if seed:
                    portrait = {"name": "user_seed", "description": "用户提供的种子公式",
                                "pseudo_code": [seed]}
                    info = {"formula": seed, "parameters": {}, "candidates": [{}]}
                else:
                    portrait = self.agent.initial_portrait(self.fields, self.avoid, self.target_text(),
                                                           self.explored_roots(tree_id))
                    info = None
            except GenerationError as error:
                self.ledger.failed(attempt, error, "initial_portrait")
                continue
            candidate = self.materialize(portrait, tree_id, None, None, None, attempt, info)
            if candidate is None:
                seed = None
                continue
            root = self.make_node(candidate, tree_id, 0)
            nodes.append(root)
            if self.review(root):
                self.search_count += 1
                self.ledger.record(attempt, "materialized", node_attempt=root.attempt_id)
                self.record_node(root)
                return root
            nodes.remove(root)
            seed = None
        raise RuntimeError(f"Tree {tree_id}: no valid root after "
                           f"{self.settings.max_root_attempts} attempts; see the experiment ledger")

    def target_text(self):
        p = self.protocol
        return (f"未来 {p['horizon']} 个交易日收益（收盘价买卖），{p['universe']} 股票池，"
                f"以截面 RankIC 衡量；训练区间 {p['train'][0]} 至 {p['train'][1]}")

    def explored_roots(self, tree_id, limit=8):
        from ..llm.brief import fmt_metrics
        roots = [r for r in self.ledger.records.values()
                 if r.get("source") == "tree_node" and r.get("parent") is None
                 and r.get("tree") != tree_id and r.get("formula")]
        return [f"{r['formula']} | {fmt_metrics(r.get('metrics'))}"
                + (f" | {'；'.join(r['reasons'])}" if r.get("reasons") else
                   " | 入库" if r.get("status") == "admitted" else "")
                for r in roots[-limit:]]

    def failure_notes(self, genes=()):
        failed = self.ledger.retrieve(genes, successes=0, rejections=0, failures=3)["execution_failed"]
        return [f"{r.get('error')}（出现 {r.get('count', 1)} 次）" for r in failed]

    def materialize(self, portrait, tree_id, dimension, parent, build, attempt, info=None):
        """Formula generation, parameter selection and the validation/correction loop."""
        genes = parent.root_genes if parent is not None else ()
        try:
            if info is None:
                info = self.agent.formula(portrait, self.fields, self.avoid, self.failure_notes(genes))
        except GenerationError as error:
            self.ledger.failed(attempt, error, "formula")
            return None
        for correction in range(self.settings.max_corrections + 1):
            try:
                return self.try_parameters(info, tree_id, dimension, parent, portrait)
            except InvalidCandidate as invalid:
                self.ledger.failed(attempt, invalid, "validation" if correction == 0 else "correction",
                                   formula=invalid.formula, parameters=invalid.parameters)
                if correction == self.settings.max_corrections:
                    return None
                key = None
                try:
                    key = structure(invalid.formula)
                except FormulaError:
                    pass
                try:
                    info = self.agent.correct(
                        portrait, invalid.formula, invalid.parameters, str(invalid), self.fields,
                        self.avoid, self.ledger.tried_parameters(key) if key else (), build)
                except GenerationError as error:
                    self.ledger.failed(attempt, error, "correction")
                    return None
                attempt = self.ledger.propose(source="correction", tree=tree_id,
                                              parent=getattr(parent, "label", None), dimension=dimension)
        return None

    def try_parameters(self, info, tree_id, dimension, parent, portrait):
        symbolic = str(info["formula"]).strip()
        try:
            names = check(symbolic, self.fields)
        except FormulaError as error:
            raise InvalidCandidate(str(error), symbolic) from None
        key, genes = structure(symbolic), extract_root_genes(symbolic)
        trials, errors = [], []
        for parameters in list(info.get("candidates") or [{}])[:3]:
            parameters = {k: v for k, v in (parameters or {}).items() if k in names}
            trial = self.ledger.propose(
                source="parameter_trial", tree=tree_id, parent=getattr(parent, "label", None),
                dimension=dimension, symbolic=symbolic, parameters=parameters, structure=key,
                root_genes=genes, description=portrait.get("description", ""))
            try:
                formula = substitute(symbolic, parameters)
                check(formula, self.fields, parameters={})
                with self.tracer.observe("evaluate-candidate", as_type="evaluator",
                                         input={"formula": formula}) as obs:
                    measurement = self.market.measure(formula, "train", self.repository)
                    obs.update(output=measurement.metrics)
            except (FormulaError, EvaluationError) as error:
                self.ledger.failed(trial, error, "evaluation")
                errors.append((parameters, str(error)))
                continue
            scores = self.reference.scores(measurement.metrics)
            self.ledger.record(trial, "evaluated", formula=formula, metrics=measurement.metrics,
                               scores=scores)
            trials.append(((float(np.mean(list(scores.values()))), measurement.metrics["rank_ir"],
                            measurement.metrics["rank_ic"]), trial, formula, parameters,
                           measurement, scores))
        if not trials:
            detail = "；".join(f"{p}: {e}" for p, e in errors) or "没有候选参数"
            raise InvalidCandidate(f"所有候选参数都无法评估：{detail}", symbolic,
                                   errors[0][0] if errors else {})
        _, trial, formula, parameters, measurement, scores = max(trials, key=lambda t: t[0])
        return {"attempt_id": trial, "formula": formula, "symbolic": symbolic,
                "parameters": parameters, "structure": key, "root_genes": extract_root_genes(formula),
                "measurement": measurement, "scores": scores, "portrait": portrait}

    def make_node(self, candidate, tree_id, index, parent=None, dimension=None):
        node = MCTSNode(candidate["formula"], parent, dimension, tree_id, f"V{index}")
        node.attempt_id = candidate["attempt_id"]
        node.symbolic, node.parameters = candidate["symbolic"], candidate["parameters"]
        node.structure, node.root_genes = candidate["structure"], candidate["root_genes"]
        node.portrait = candidate["portrait"]
        node.metrics = candidate["measurement"].metrics
        node.correlations = candidate["measurement"].correlations
        node.ranked = candidate["measurement"].ranked
        node.scores = dict(candidate["scores"])
        node.score = None
        if parent is not None:
            parent.children.append(node)
        return node

    def review(self, node):
        """LLM overfitting score from the formula, its lineage and neighbouring branches."""
        neighbors = [edge_line(n) for n in node.siblings()] + [edge_line(n) for n in node.children]
        try:
            score, reason = self.agent.review_overfitting(node.formula, history_lines(node), neighbors)
        except GenerationError as error:
            self.ledger.failed(node.attempt_id, error, "overfitting_review")
            return False
        node.scores["Overfitting"] = score
        node.overfit_reason = reason
        node.score = float(np.mean([node.scores[d] for d in DIMENSIONS]))
        return True

    def record_node(self, node):
        self.ledger.record(
            node.attempt_id, "evaluated", source="tree_node", tree=node.tree, label=node.label,
            parent=node.parent.label if node.parent else None,
            parent_attempt=node.parent.attempt_id if node.parent else None,
            dimension=node.dimension, formula=node.formula, symbolic=node.symbolic,
            parameters=node.parameters, structure=node.structure, root_genes=node.root_genes,
            description=node.portrait.get("description", ""), suggestions=node.suggestions,
            metrics=node.metrics, scores=node.scores, reward=node.score,
            overfit_reason=node.overfit_reason, summary=node.summary,
            repository_snapshot=self.repository.snapshot_ids())

    # --------------------------------------------------------------- admission
    def admit(self, nodes):
        """Appendix F.1: check every node of the finished tree, in creation order."""
        with self.tracer.observe("decide-repository-admission", as_type="evaluator",
                                 input={"nodes": [n.label for n in nodes]}) as obs:
            decisions = {}
            for node in nodes:
                if self.repository.contains(node.formula):
                    reasons = ["与有效因子库中已有公式相同"]
                    correlations = {}
                else:
                    correlations = dict(node.correlations)
                    for alpha in self.repository:
                        if alpha.attempt_id not in correlations:
                            correlations[alpha.attempt_id] = rank_correlation(node.ranked, alpha.ranked)
                    metrics = dict(node.metrics, max_corr=max(correlations.values(), default=0.0))
                    reasons = self.criteria.failures(metrics, self.repository)
                if reasons:
                    node.admission = "未入库（" + "；".join(reasons) + "）"
                    self.ledger.record(node.attempt_id, "rejected", reasons=reasons)
                else:
                    self.repository.add(RepositoryAlpha(
                        node.attempt_id, node.formula, dict(node.metrics), node.ranked, node.tree,
                        node.label, node.summary, node.portrait.get("description", "")), correlations)
                    node.admission = "入库"
                    self.ledger.record(node.attempt_id, "admitted",
                                       repository_size=len(self.repository))
                decisions[node.label] = node.admission
                print(f"[{node.name}] {node.admission}", flush=True)
            for node in nodes:
                node.ranked = None          # the repository keeps admitted signals
            self.fsa.fit([a.formula for a in self.repository])
            self.avoid = self.fsa.avoid_patterns()
            obs.update(output={"decisions": decisions, "avoid": self.avoid,
                               "repository_size": len(self.repository)})

    # ------------------------------------------------------------- persistence
    def save_checkpoint(self):
        if not self.settings.checkpoint_path:
            return None
        path = Path(self.settings.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"repository": self.repository, "ledger": self.ledger.snapshot(),
                 "search_count": self.search_count, "tree_index": self.tree_index,
                 "avoid": self.avoid, "trees": self.trees, "rng": self.rng.bit_generator.state,
                 "settings": asdict(self.settings)}
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(state, stream)
        temporary.replace(path)
        return path

    def load_checkpoint(self, path):
        with open(path, "rb") as stream:
            state = pickle.load(stream)
        self.ledger.restore(state["ledger"])
        self.repository, self.search_count = state["repository"], state["search_count"]
        self.tree_index, self.avoid, self.trees = state["tree_index"], state["avoid"], state["trees"]
        self.rng.bit_generator.state = state["rng"]
        self.fsa.fit([a.formula for a in self.repository])

    def final_alphas(self, k):
        """Top-k repository alphas by training RankIR (Appendix F.1)."""
        return sorted(self.repository, key=lambda a: a.metrics["rank_ir"], reverse=True)[:k]
