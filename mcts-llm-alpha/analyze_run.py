#!/usr/bin/env python
"""Summarise a finished run directory without further model calls.

Reports search shape, admissions and rejection reasons, dimension choices, how
often the stated intent of a refinement was met, score ties, and LLM latency
and token usage per role (from llm_calls.jsonl when present).
"""
import argparse
import json
from collections import Counter, defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mcts_llm_alpha.evaluation.protocol import LOWER_IS_BETTER, METRIC_OF  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    folder = parser.parse_args().run_directory.resolve()
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    nodes = json.loads((folder / "search_trees.json").read_text(encoding="utf-8"))
    by_key = {(n["tree"], n["label"]): n for n in nodes}

    reasons, dims, intents = Counter(), Counter(), Counter()
    for n in nodes:
        if n["admission"] and n["admission"] != "入库":
            for part in n["admission"].strip("未入库（）").split("；"):
                reasons[part.split(" ")[0]] += 1
        if n["dimension"]:
            dims[n["dimension"]] += 1
            parent = by_key[(n["tree"], n["parent"])]
            for dim, direction in ((n.get("suggestions") or {}).get("expected_effect") or {}).items():
                metric = METRIC_OF.get(dim)
                if metric:     # data dimensions are judged on the raw metric
                    change = n["metrics"][metric] - parent["metrics"][metric]
                    change = -change if metric in LOWER_IS_BETTER else change
                elif dim in n["scores"] and dim in parent["scores"]:
                    change = n["scores"][dim] - parent["scores"][dim]
                else:
                    continue
                up = str(direction).strip().startswith("+")
                intents["met" if (change > 0 if up else change < 0) else
                        "tie" if change == 0 else "missed"] += 1
    trees = defaultdict(list)
    for n in nodes:
        trees[n["tree"]].append(n)
    summary = {
        "status": manifest.get("status"), "search_count": manifest.get("search_count"),
        "trees": len(trees), "nodes_per_tree": [len(v) for v in trees.values()],
        "repository_size": manifest.get("repository_size"),
        "admitted": [n["formula"] for n in nodes if n["admission"] == "入库"],
        "rejection_reasons": dict(reasons), "dimension_choices": dict(dims),
        "intent_vs_outcome": dict(intents),
        "tied_tree_scores": sum(len(v) - len({round(n["score"], 6) for n in v}) for v in trees.values()),
        "best_rank_ic": max(nodes, key=lambda n: n["metrics"]["rank_ic"])["formula"] if nodes else None,
        "ledger_statuses": manifest.get("ledger_statuses"),
    }
    calls = folder / "llm_calls.jsonl"
    if calls.exists():
        roles = defaultdict(lambda: {"calls": 0, "errors": 0, "latency_s": 0.0, "output_tokens": 0})
        for line in calls.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            r = roles[row["name"]]
            r["calls"] += 1
            r["errors"] += bool(row.get("error"))
            r["latency_s"] += row.get("latency_s") or 0
            r["output_tokens"] += ((row.get("usage") or {}).get("completion_tokens") or 0)
        summary["llm_roles"] = {k: {**v, "latency_s": round(v["latency_s"], 1)} for k, v in roles.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (folder / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
