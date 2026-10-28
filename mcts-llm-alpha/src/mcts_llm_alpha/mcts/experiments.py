"""Append-only experiment ledger: what was tried, what was observed, what was decided.

Every candidate (tree node or parameter trial) gets one attempt record built from
events. Retrieval for the LLM is structural: formulas are compared by their
shared root genes (parameter-free subtrees), near-duplicates are folded into one
row with a count, and execution failures are grouped by error message. A failed
execution is never presented as evidence of a weak economic signal.
"""

import hashlib
import json
import math
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = 2
SUCCESS, REJECTED, FAILED = "admitted", "rejected", "execution_failed"


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        return _safe(value.item())
    return str(value)


def error_kind(message):
    """Group error messages that differ only in numbers or quoted formulas."""
    text = re.sub(r"\d+(\.\d+)?%?", "#", str(message or ""))
    return re.sub(r"\s+", " ", text)[:120]


class ExperimentLedger:
    def __init__(self, path=None, evaluation_context=None):
        self.path = Path(path) if path else None
        self.context = _safe(evaluation_context or {"scope": "unspecified"})
        encoded = json.dumps(self.context, sort_keys=True, ensure_ascii=False)
        self.scope_id = hashlib.sha256(encoded.encode()).hexdigest()[:20]
        self.run_id = uuid4().hex
        self.records = OrderedDict()
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("scope_id") == self.scope_id and row.get("schema_version") == SCHEMA_VERSION:
                    self._apply(row)

    def _apply(self, event):
        record = self.records.setdefault(event["attempt_id"], {"attempt_id": event["attempt_id"]})
        record.update(event["details"])
        record.update(status=event["event"], updated_at=event["timestamp"])

    def record(self, attempt_id, event, **details):
        row = {"schema_version": SCHEMA_VERSION, "scope_id": self.scope_id, "run_id": self.run_id,
               "attempt_id": attempt_id, "event": event,
               "timestamp": datetime.now(timezone.utc).isoformat(), "details": _safe(details)}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._apply(row)
        return attempt_id

    def propose(self, *, formula="", source, **details):
        return self.record(uuid4().hex, "proposed", formula=formula, source=source, **details)

    def failed(self, attempt_id, error, stage, **details):
        self.record(attempt_id, FAILED, stage=stage, error=str(error), error_kind=error_kind(error),
                    **details)

    def retrieve(self, root_genes, *, exclude=(), dimension=None, successes=4, rejections=4,
                 failures=3):
        """Structurally similar outcomes, folded by structure, grouped by kind."""
        genes, exclude = set(root_genes), set(exclude)

        def similarity(record):
            other = set(record.get("root_genes") or ())
            score = len(genes & other) / max(len(genes | other), 1)
            return score + (0.1 if dimension and record.get("dimension") == dimension else 0.0)

        folded = {}
        for record in self.records.values():
            if record["attempt_id"] in exclude or record.get("source") == "parameter_trial":
                continue
            status = record.get("status")
            kind = SUCCESS if status == SUCCESS else REJECTED if status == REJECTED else \
                FAILED if status == FAILED else None
            if kind is None:
                continue
            key = (kind, record.get("error_kind") if kind == FAILED else record.get("structure"))
            entry = folded.get(key)
            if entry is None:
                folded[key] = entry = {"record": record, "count": 0, "similarity": 0.0}
            entry["count"] += 1
            entry["similarity"] = max(entry["similarity"], similarity(record))
            if kind != FAILED and (record.get("reward") or 0) > (entry["record"].get("reward") or 0):
                entry["record"] = record
        result = {}
        for kind, limit in ((SUCCESS, successes), (REJECTED, rejections), (FAILED, failures)):
            rows = [e for (k, _), e in folded.items() if k == kind]
            rows.sort(key=lambda e: (e["similarity"], e["count"], e["record"].get("updated_at", "")),
                      reverse=True)
            result[kind] = [dict(e["record"], count=e["count"]) for e in rows[:limit]]
        return result

    def tried_parameters(self, structure_key, limit=6):
        rows = [r for r in self.records.values()
                if r.get("structure") == structure_key and r.get("parameters")]
        return [{"parameters": r["parameters"], "status": r.get("status"),
                 "metrics": r.get("metrics"), "error": r.get("error")} for r in rows[-limit:]]

    def snapshot(self):
        return {"scope_id": self.scope_id, "run_id": self.run_id, "records": _safe(self.records)}

    def restore(self, snapshot):
        if snapshot["scope_id"] != self.scope_id:
            raise ValueError("Checkpoint evaluation protocol differs from the experiment ledger")
        self.run_id = snapshot["run_id"]
        self.records = OrderedDict(snapshot["records"])
