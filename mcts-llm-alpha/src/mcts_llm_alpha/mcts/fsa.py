"""Frequent Subtree Avoidance (Section 4.5).

A root gene is a subtree of the expression tree that grows upward from the raw
features, with every operator parameter abstracted to ``t``. The miner counts, for
the alphas of the effective repository, in how many alphas each root gene occurs,
keeps the closed ones (no immediate supertree has the same support) and returns
the top-k most frequent for the LLM to avoid.
"""

import ast
from collections import defaultdict

from ..formula import parse, render


def _has_field(node):
    return any(isinstance(n, ast.Name) and n.id.startswith("field_") for n in ast.walk(node))


def _children(node):
    if isinstance(node, ast.Call):
        return list(node.args)
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    return []


def root_gene_tree(formula):
    """Map each root gene of ``formula`` to the root genes directly above it."""
    tree = parse(formula).body
    parents = defaultdict(set)

    def visit(node, parent_pattern):
        if not _children(node) or not _has_field(node):
            return
        pattern = render(node, abstract=True)
        parents[pattern]
        if parent_pattern:
            parents[pattern].add(parent_pattern)
        for child in _children(node):
            visit(child, pattern)

    visit(tree, None)
    return dict(parents)


def extract_root_genes(formula):
    return sorted(root_gene_tree(formula))


class FrequentSubtreeMiner:
    def __init__(self, top_k=3, min_support=2):
        self.top_k, self.min_support = top_k, min_support
        self.support = {}
        self.supertrees = defaultdict(set)

    def fit(self, formulas):
        support = defaultdict(int)
        self.supertrees = defaultdict(set)
        for formula in formulas:
            genes = root_gene_tree(formula)
            for gene, parents in genes.items():
                support[gene] += 1
                self.supertrees[gene] |= parents
        self.support = dict(support)
        return self

    def closed_frequent(self):
        frequent = {g: c for g, c in self.support.items() if c >= self.min_support}
        closed = [g for g, c in frequent.items()
                  if not any(self.support.get(p) == c for p in self.supertrees[g])]
        return sorted(closed, key=lambda g: (-frequent[g], -len(g), g))

    def avoid_patterns(self):
        return self.closed_frequent()[:self.top_k]
