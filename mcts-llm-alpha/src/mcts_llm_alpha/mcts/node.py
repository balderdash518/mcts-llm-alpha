"""A search-tree node: one evaluated alpha plus the refinement that produced it."""

import math

from ..evaluation.protocol import DIMENSIONS


class MCTSNode:
    DIMENSIONS = DIMENSIONS

    def __init__(self, formula, parent=None, dimension=None, tree=0, label="V0"):
        self.formula = formula            # concrete formula, paper syntax
        self.parent = parent
        self.dimension = dimension        # dimension refined from the parent
        self.tree, self.label = tree, label
        self.children = []
        self.visits, self.q = 0, 0.0
        self.symbolic = formula
        self.parameters = {}
        self.structure = None
        self.root_genes = []
        self.portrait = {}
        self.suggestions = {}
        self.metrics = {}
        self.scores = {}
        self.score = 0.0
        self.overfit_reason = ""
        self.summary = ""
        self.correlations = {}
        self.ranked = None
        self.attempt_id = None
        self.admission = None             # decided when the tree is finished

    @property
    def name(self):
        return f"T{self.tree}-{self.label}"

    def lineage(self):
        nodes, current = [], self
        while current is not None:
            nodes.append(current)
            current = current.parent
        return nodes[::-1]

    def siblings(self):
        return [n for n in self.parent.children if n is not self] if self.parent else []

    def uct(self, c):
        """Eq. (1): Q(s, a) + c * sqrt(ln N_s / N_s')."""
        return self.q + c * math.sqrt(math.log(max(self.parent.visits, 1)) / self.visits)

    def virtual_uct(self, c):
        """Virtual action that expands this node; its visit count is the number of children."""
        return self.q + c * math.sqrt(math.log(max(self.visits, 1)) / len(self.children))

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()
