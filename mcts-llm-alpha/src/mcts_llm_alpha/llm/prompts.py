"""Prompt templates (Appendix J) plus the refinement-suggestion and correction prompts.

Flow of one expansion: suggestion prompt (reads the SearchBrief and few-shot
examples) -> refinement prompt J.4 (applies the suggestions) -> formula prompt
J.2 (symbolic formula and three parameter candidates) -> correction prompt while
the formula is invalid -> overfitting review J.3 -> refinement summary.
"""

from ..formula import fields_text, operators_text
from .brief import fmt_metrics

DIMENSION_FOCUS = {
    "Effectiveness": "提高预测能力，即提升训练期 RankIC。",
    "Stability": "提高预测能力的稳定性，即提升 RankIR（日 RankIC 的均值/标准差）。",
    "Turnover": "降低日换手率，使因子在相邻交易日的截面持仓变化更小，同时不牺牲预测能力。",
    "Diversity": "降低与有效因子库中已有因子的相关性，引入新的信息来源或结构。",
    "Overfitting": "降低过拟合风险：让结构有清晰的经济动机，避免无依据的复杂化和参数微调。",
}

_PSEUDO_RULES = """1. Alpha 值应为无量纲（无单位）。
2. 所有回看窗口和其他数值参数都必须在伪代码中写成具名参数，遵循 Python 命名规范（如 lookback_period, volatility_window）。
3. Alpha 的参数总数不得超过 3 个。
4. 伪代码应逐步体现计算过程，只使用“可用算子”和已定义的参数，每行只做一次运算。
5. 伪代码中使用能说明含义的变量名。"""

_PORTRAIT_FORMAT = """输出必须是 JSON，包含三个键：
1. "name"：Alpha 名称（Python 变量命名风格，如 price_volatility_ratio）。
2. "description"：简明说明该 Alpha 的用途和背后的直观动机，避免过于技术化。
3. "pseudo_code"：字符串列表，每行一个运算，格式为 变量名 = 算子名(input=[输入1, ...], param=[参数1, ...])，输入只能是数据字段或之前计算的变量；四则运算、取负可直接写成表达式（如 c = a * b）。

格式示例：
{"name": "volatility_adjusted_momentum", "description": "……", "pseudo_code": ["……"]}"""


def _avoid(avoid):
    return "、".join(avoid) if avoid else "无"


def _common(fields):
    return f"""可用数据字段：
{fields_text(fields)}

可用算子：
{operators_text()}"""


def initial_portrait_prompt(fields, avoid, target="", explored=()):
    """Appendix J.1: alpha portrait generation, plus the roots of earlier trees."""
    target_text = f"\n预测目标：{target}\n" if target else ""
    explored_text = ""
    if explored:
        explored_text = ("\n此前各搜索树的根因子（请提出不同的经济假设，不要换个窗口重复它们）：\n"
                         + "\n".join(f"- {e}" for e in explored) + "\n")
    return f"""任务描述：
你是一位专注于因子投资的量化金融专家。请设计一个用于投资策略的 Alpha 因子，并按指定格式输出。
{target_text}
{_common(fields)}
{explored_text}
Alpha 要求：
{_PSEUDO_RULES}
6. Alpha 公式至少包含两种不同的可用算子。
7. 设计 Alpha 表达式时，尽量避免包含以下子表达式：{_avoid(avoid)}

格式要求：
{_PORTRAIT_FORMAT}"""


def formula_prompt(portrait, fields, avoid, notes=()):
    """Appendix J.2: from portrait to a symbolic formula with three parameter candidates."""
    notes_text = ("\n\n已知的执行问题（请避免重蹈覆辙）：\n" + "\n".join(f"- {n}" for n in notes)) if notes else ""
    return f"""任务描述：
根据下面的 Alpha 画像（名称、说明、伪代码），写出对应的 Alpha 公式。

{_common(fields)}

Alpha 画像：
名称：{portrait.get('name', '')}
说明：{portrait.get('description', '')}
伪代码：
{chr(10).join(portrait.get('pseudo_code', []))}

公式要求：
1. 只使用上面的字段和算子，字段必须带 $（如 $close），算子名称与列表完全一致。
2. 数值参数用伪代码里的参数名表示（如 lookback_period），不要直接写数值；参数总数不超过 3 个。
3. 窗口参数必须是正整数；禁止引用未来数据。
4. 注意定义域：Log 的输入必须为正，分母不能恒为 0。
5. 设计时尽量避免包含以下子表达式：{_avoid(avoid)}{notes_text}

格式要求：
只输出 JSON：
{{
  "formula": "使用参数名的公式",
  "parameters": {{"参数名": {{"description": "含义", "range": [最小值, 最大值]}}}},
  "candidates": [{{"参数名": 值}}, {{"参数名": 值}}, {{"参数名": 值}}]
}}
"candidates" 必须给出 3 组不同的候选参数。"""


def suggestion_prompt(brief, dimension, examples, avoid, fields):
    """Refinement suggestion generation (Section 4.2, Appendix C.2)."""
    if examples:
        example_text = "\n".join(
            f"- {e['reason']}：{e['formula']} | {fmt_metrics(e['metrics'])}"
            + (f" | {e['description']}" if e.get('description') else "") for e in examples)
        example_text = "\n有效因子库中的参考示例（few-shot）：\n" + example_text
    else:
        example_text = "\n本维度不提供示例（zero-shot）。"
    return f"""任务描述：
你是量化因子研究员。下面是一个 Alpha 因子当前的评估结果、它的改进谱系、相邻分支的尝试，以及历史实验。请针对维度 {dimension} 给出改进建议。

改进目标：{DIMENSION_FOCUS[dimension]}

{_common(fields)}

{brief}
{example_text}

要求：
1. 先根据谱系和相邻分支诊断：哪些改动确实改善了目标指标，哪些没有，以及当前公式在 {dimension} 上偏弱的可能原因。
2. 给出 3 条彼此不同、可直接落实到公式上的建议。不要重复“当前节点已尝试过的改进”和历史失败中已经证明无效的方向；可以是小幅修改，也可以是更大的结构替换，只要理由充分。
3. 每条建议说明为什么预期有效，以及对其他维度可能的影响；建议中的公式只能使用上面列出的字段和算子。
4. 建议不应引入以下高频子结构：{_avoid(avoid)}

只输出 JSON：
{{
  "diagnosis": "一到三句诊断",
  "suggestions": ["建议1", "建议2", "建议3"],
  "avoid_repeating": ["本次不应再重复的已尝试方向"],
  "expected_effect": {{"{dimension}": "+", "其他维度名": "+ 或 -"}}
}}
expected_effect 只填写你有把握的维度；"+" 表示分数预期上升，"-" 表示预期下降。"""


def refinement_prompt(formula, suggestions, fields, avoid):
    """Appendix J.4: apply refinement suggestions to the original alpha."""
    lines = "\n".join(f"- {s}" for s in suggestions.get("suggestions", []))
    if suggestions.get("diagnosis"):
        lines = f"诊断：{suggestions['diagnosis']}\n" + lines
    return f"""任务描述：
有一个用于量化投资、预测资产价格趋势的 Alpha 因子。请根据下面的建议改进它，并给出改进后的 Alpha。

{_common(fields)}

Alpha 要求：
{_PSEUDO_RULES}
6. 设计 Alpha 表达式时，尽量避免包含以下子表达式：{_avoid(avoid)}

原始 Alpha 表达式：
{formula}

改进建议：
注意：以下建议不必全部采纳，只需斟酌并采纳其中合理的部分。
{lines}

格式要求：
{_PORTRAIT_FORMAT}"""


def correction_prompt(portrait, formula, parameters, error, fields, avoid, tried=(), brief=""):
    """CorrectAlphaFormula (Algorithm 1, line 19): fix an invalid formula using the feedback."""
    tried_text = ""
    if tried:
        tried_text = "\n同一结构已试过的参数：\n" + "\n".join(
            f"- {t['parameters']}：" + (f"错误 {t['error']}" if t.get('error') else fmt_metrics(t.get('metrics')))
            for t in tried)
    brief_text = f"\n\n搜索上下文：\n{brief}" if brief else ""
    return f"""任务描述：
下面的 Alpha 公式没有通过校验或无法在行情数据上计算。请根据错误信息修正公式，尽量保留画像描述的经济含义。

{_common(fields)}

Alpha 画像：
名称：{portrait.get('name', '')}
说明：{portrait.get('description', '')}
伪代码：
{chr(10).join(portrait.get('pseudo_code', []))}

出错的公式：{formula}
参数取值：{parameters}
错误信息：{error}{tried_text}{brief_text}

修正要求：
1. 只使用上面的字段和算子；数值参数用参数名表示，参数总数不超过 3 个，窗口为正整数。
2. 针对错误信息修改，不要无关地改动其余结构。
3. 设计时尽量避免包含以下子表达式：{_avoid(avoid)}

只输出 JSON：
{{
  "formula": "修正后的公式（使用参数名）",
  "parameters": {{"参数名": {{"description": "含义", "range": [最小值, 最大值]}}}},
  "candidates": [{{"参数名": 值}}, {{"参数名": 值}}, {{"参数名": 值}}]
}}"""


def summary_prompt(parent, child, dimension, deltas, score_changes):
    """GenerateRefinementSummary (Algorithm 1, line 28)."""
    applied = "；".join((child.suggestions or {}).get("suggestions", []))
    changes = "，".join(f"{d} {v:+.2f}" for d, v in score_changes.items())
    return f"""请为下面这一步 Alpha 改进写一个简洁总结（1-2 句话），说明具体改了什么、目标维度是否改善、对其他维度有什么影响。只陈述观察到的变化，不要推断普遍规律。
说明：五个维度分数都是越高越好（Overfitting 分越高表示过拟合风险越低），分数是相对参照集的排名分；原始指标中 RankIC、RankIR 越高越好，日换手率、最大相关越低越好。

改进维度：{dimension}
原公式：{parent.formula}
新公式：{child.formula}
采用的建议：{applied or '无'}
新因子说明：{child.portrait.get('description', '')}
原始指标变化：{deltas or '无'}
维度分数变化：{changes}

直接输出总结，不要任何格式。"""


def overfitting_prompt(formula, history, neighbors):
    """Appendix J.3: critical overfitting risk assessment."""
    neighbor_text = "\n".join(neighbors) if neighbors else "无"
    operators = operators_text()
    return f"""任务：Alpha 过拟合风险的批判性评估

请根据下面 Alpha 的表达式及其改进历史，严格评估其过拟合风险与泛化能力，重点关注复杂度和优化过程是否有合理依据，还是存在过度拟合的迹象。

输入：
- Alpha 表达式：{formula}
- 改进历史（根→当前，每步含改动摘要与指标变化）：
{chr(10).join(history)}
- 搜索树中相邻分支的尝试（同一父节点的其他改进与该因子已有的子节点）：
{neighbor_text}

算子含义：
{operators}
历史中的 [改进 X] 表示该步针对的目标维度，不代表该步被判定为过拟合。

评估标准：
1. 合理动机 vs. 复杂度：表达式的复杂度是否有经济动机支撑，还是显得随意、冗余，像在拟合噪声？
2. 有原则的开发 vs. 数据挖掘：改进历史是基于假设的逐步改进，还是频繁、缺乏解释的参数微调？相邻分支是否显示在同一处反复试错？
3. 透明度 vs. 不透明：逻辑是否容易理解（即使较复杂），还是不透明到可能掩盖过拟合？

打分与输出：
- 给出 0-10 的分数：10 表示过拟合风险极低（对泛化能力高度有信心），0 表示风险极高。请充分利用整个区间。
- 用一句话说明打分理由，指出关键依据。
- 只输出 JSON：{{"reason": "理由", "score": 数字}}"""
