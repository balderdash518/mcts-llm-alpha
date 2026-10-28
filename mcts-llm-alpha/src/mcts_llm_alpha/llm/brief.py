"""SearchBrief: the structured evidence the LLM reads before refining an alpha.

Sections, in priority order: evaluation protocol, current alpha card, lineage
(each edge once: what was changed, how the raw metrics moved, whether the stated
intent was met), neighbourhood (refinements already tried from the parent and
from this node), structurally similar experiments from earlier trees, and
constraints. Lower-priority sections shrink first when the context is short;
no identifiers, timestamps or signal matrices are shown.
"""

from ..evaluation.protocol import DIMENSIONS, LOWER_IS_BETTER, METRIC_OF, METRIC_TEXT

LEVELS = (
    dict(lineage=12, neighbors=8, success=4, rejected=4, failed=3),
    dict(lineage=6, neighbors=4, success=2, rejected=2, failed=2),
    dict(lineage=3, neighbors=2, success=1, rejected=1, failed=1),
    dict(lineage=1, neighbors=0, success=0, rejected=0, failed=0),
)
SHORT = {"Effectiveness": "Eff", "Stability": "Stab", "Turnover": "Turn",
         "Diversity": "Div", "Overfitting": "OR"}
SHOWN_METRICS = ("rank_ic", "rank_ir", "turnover", "max_corr")
STATUS_TEXT = {"admitted": "入库", "rejected": "未入库", "execution_failed": "执行失败"}


def fmt_metrics(metrics):
    if not metrics:
        return "未测量"
    digits = {"rank_ic": 4, "rank_ir": 3, "turnover": 3, "max_corr": 3}
    return "，".join(f"{METRIC_TEXT[k]} {metrics[k]:.{digits[k]}f}" for k in SHOWN_METRICS if k in metrics)


def fmt_scores(scores, total=None):
    if not scores:
        return "未评分"
    text = " ".join(f"{SHORT[d]} {scores[d]:.1f}" for d in DIMENSIONS if d in scores)
    return text + (f"（总分 {total:.2f}）" if total is not None else "")


def fmt_deltas(child, parent):
    if not child.metrics or not parent.metrics:
        return ""
    digits = {"rank_ic": 4, "rank_ir": 3, "turnover": 3, "max_corr": 3}
    return "，".join(f"Δ{METRIC_TEXT[k]} {child.metrics[k] - parent.metrics[k]:+.{digits[k]}f}"
                    for k in SHOWN_METRICS if k in child.metrics and k in parent.metrics)


def intent_outcomes(node):
    """(dimension, wanted up?, observed change text, met?) for each stated intent.

    Data dimensions are judged on the raw metric: with a small reference set the
    rank scores are coarse and often do not move even when the metric does.
    """
    expected = (node.suggestions or {}).get("expected_effect") or {}
    if node.parent is None or not isinstance(expected, dict):
        return []
    outcomes = []
    for dim, direction in expected.items():
        if dim not in SHORT:
            continue
        want_up = str(direction).strip().startswith(("+", "↑", "up", "提高", "上升"))
        metric = METRIC_OF.get(dim)
        if metric and metric in node.metrics and metric in node.parent.metrics:
            change = node.metrics[metric] - node.parent.metrics[metric]
            better = change < 0 if metric in LOWER_IS_BETTER else change > 0
            text = f"{METRIC_TEXT[metric]} {change:+.4f}"
        elif dim in node.scores and dim in node.parent.scores:
            change = node.scores[dim] - node.parent.scores[dim]
            better, text = change > 0, f"分数 {change:+.1f}"
        else:
            continue
        outcomes.append((dim, want_up, text, change != 0 and better == want_up))
    return outcomes


def fmt_expectation(node):
    """Compare the refinement's stated intent with what was observed."""
    parts = [f"{SHORT[d]}{'↑' if up else '↓'}→{text}{'达成' if met else '未达成'}"
             for d, up, text, met in intent_outcomes(node)]
    return ("；预期 " + " ".join(parts)) if parts else ""


def fmt_admission(node):
    if not node.admission:
        return ""
    return "；结论：" + node.admission


def edge_line(node):
    parent = node.parent
    what = node.summary or node.portrait.get("description", "")
    total = f" | 总分 {parent.score:.2f}→{node.score:.2f}" if node.score is not None else ""
    return (f"{parent.label}→{node.label} [改进 {node.dimension}] {what} | {fmt_deltas(node, parent)}"
            f"{total}{fmt_expectation(node)}{fmt_admission(node)}")


def protocol_lines(protocol, repository_size, reference_size=None):
    c = protocol["criteria"]
    return [
        "【评估协议】",
        f"- 预测目标：未来 {protocol['horizon']} 个交易日收益（收盘价买卖）；训练区间 "
        f"{protocol['train'][0]} 至 {protocol['train'][1]}；股票池 {protocol['universe']}（按日成分）。"
        "测试区间不参与搜索。",
        "- Effectiveness=RankIC（越高越好）；Stability=RankIR=日 RankIC 均值/标准差（越高越好）；"
        "Turnover=日换手率 Σ|w_t−w_{t−1}|，范围 0–2（越低越好）；Diversity=与有效因子库因子的最大相关"
        "（越低越好，库为空时为 0）；Overfitting=LLM 审阅的过拟合风险分。",
        "- 五个维度分数都是越高越好（Overfitting 分越高表示过拟合风险越低），总分为五维均值；"
        "谱系中的 [改进 X] 表示该步针对的目标维度。",
        f"- 前四个维度分数=10×(1−参照集中优于该因子的比例)，当前有效因子库 {repository_size} 个；"
        + (f"库为空，暂以此前已评估的 {reference_size} 个候选为参照。" if not repository_size and reference_size
           else "库为空且无参照时这四维都是 10。" if not repository_size else "参照集即有效因子库。")
        + "维度分数只在同一参照快照下可比，原始指标才是绝对水平。",
        f"- 入库标准：RankIC≥{c['rank_ic_min']}，RankIR≥{c['rank_ir_min']}，"
        f"R_RankIC≤{c['relative_rank_max']}，R_RankIR≤{c['relative_rank_max']}"
        f"（库中因子达到 {c['relative_rank_min_size']} 个后启用），"
        f"日换手率≤{c['turnover_max']}，最大相关<{c['max_correlation']}。",
    ]


def node_card(node, dimension=None, probabilities=None):
    lines = [f"【当前因子 {node.label}】", f"公式：{node.formula}"]
    if node.symbolic != node.formula:
        lines.append(f"符号形式：{node.symbolic}；参数：{node.parameters}")
    if node.portrait.get("description"):
        lines.append(f"经济含义：{node.portrait['description']}")
    lines.append(f"原始指标：{fmt_metrics(node.metrics)}")
    if node.metrics.get("rank_ic", 0) < 0:
        lines.append("注意：RankIC 为负，表示因子方向与未来收益相反；入库标准按带符号的 RankIC/RankIR 判定。")
    lines.append(f"维度分数：{fmt_scores(node.scores, node.score)}")
    if node.overfit_reason:
        lines.append(f"过拟合审阅：{node.overfit_reason}")
    if dimension:
        prob = ""
        if probabilities:
            prob = "（抽样概率：" + " ".join(f"{SHORT[d]} {p:.2f}" for d, p in probabilities.items()) + "）"
        metric = METRIC_OF.get(dimension)
        target = f"，对应指标 {METRIC_TEXT[metric]}" if metric else "，由 LLM 过拟合审阅评分"
        lines.append(f"本次改进维度：{dimension}{target}{prob}")
    return lines


def lineage_lines(node, limit):
    edges = node.lineage()[1:]
    if not edges or limit <= 0:
        return []
    lines = ["【改进谱系（根→当前，每步一次）】",
             f"{node.lineage()[0].label} 根节点：{node.lineage()[0].formula} | "
             f"{fmt_metrics(node.lineage()[0].metrics)}"]
    hidden = edges[:-limit] if len(edges) > limit else []
    if hidden:
        dims = "、".join(n.dimension for n in hidden)
        lines.append(f"（较早 {len(hidden)} 步已折叠：维度依次为 {dims}；"
                     f"总分 {hidden[0].parent.score:.2f}→{hidden[-1].score:.2f}）")
    lines += [edge_line(n) for n in edges[-limit:]]
    return lines


def neighborhood_lines(node, limit):
    if limit <= 0:
        return []
    lines = []
    children = node.children[-limit:]
    if children:
        lines.append("【当前节点已尝试过的改进（避免重复）】")
        lines += [edge_line(child) for child in children]
    siblings = node.siblings()[-limit:]
    if siblings:
        lines.append(f"【父节点 {node.parent.label} 的其他改进分支】")
        lines += [edge_line(s) for s in siblings]
    return lines


def experience_lines(retrieved, level):
    if not retrieved:
        return []
    lines = []
    for kind, title in (("admitted", "结构相近、已入库的因子"),
                        ("rejected", "结构相近、未能入库的因子"),
                        ("execution_failed", "执行失败（按错误归类）")):
        rows = retrieved.get(kind, [])[:level[{"admitted": "success", "rejected": "rejected",
                                               "execution_failed": "failed"}[kind]]]
        if not rows:
            continue
        lines.append(f"【历史实验：{title}】")
        for r in rows:
            count = f" ×{r['count']}" if r.get("count", 1) > 1 else ""
            where = f"T{r.get('tree')}-{r.get('label')}" if r.get("label") else r.get("source", "")
            if kind == "execution_failed":
                lines.append(f"- [{r.get('stage')}] {r.get('error')}{count}；例：{r.get('formula') or '无公式'}")
            else:
                reasons = "；".join(r.get("reasons") or [])
                lines.append(f"- {where} {r.get('formula')} | {fmt_metrics(r.get('metrics'))}"
                             + (f" | 原因：{reasons}" if reasons else "") + count)
    return lines


def constraint_lines(avoid, last_error=None):
    lines = ["【约束】", "避免出现的高频子结构：" + ("、".join(avoid) if avoid else "无")]
    if last_error:
        lines.append(f"上一次尝试的错误：{last_error}")
    return lines


def render_brief(node, protocol, repository_size, reference_size=None, *, dimension=None,
                 probabilities=None, retrieved=None, avoid=(), last_error=None, level=0):
    budget = LEVELS[min(level, len(LEVELS) - 1)]
    sections = [protocol_lines(protocol, repository_size, reference_size),
                node_card(node, dimension, probabilities),
                lineage_lines(node, budget["lineage"]),
                neighborhood_lines(node, budget["neighbors"]),
                experience_lines(retrieved, budget),
                constraint_lines(avoid, last_error)]
    return "\n".join("\n".join(s) for s in sections if s)


def history_lines(node):
    """Refinement history for the overfitting review (Appendix J.3)."""
    edges = node.lineage()[1:]
    if not edges:
        return ["无改进历史（初始公式）"]
    return [edge_line(n) for n in edges]
