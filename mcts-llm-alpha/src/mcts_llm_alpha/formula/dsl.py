"""The paper's alpha operator language (Appendix D, Table 2) and its Qlib translation.

Formulas are written with the paper's operator names. They are parsed, checked
and rendered through one AST so that validation errors can be returned to the
LLM verbatim, and so that the same canonical form feeds deduplication and FSA.
"""

import ast
import math
import re

ALL_FIELDS = ('open', 'high', 'low', 'close', 'volume', 'vwap')

# name -> (arity, {argument index: minimum window}, description shown to the LLM)
OPERATORS = {
    'Abs': (1, {}, '绝对值 |x|'),
    'Sign': (1, {}, 'x 的符号'),
    'Sin': (1, {}, '正弦'),
    'Cos': (1, {}, '余弦'),
    'Tanh': (1, {}, '双曲正切'),
    'Log': (1, {}, '自然对数'),
    'Delay': (2, {1: 1}, 'Delay(x, t)：x 在 t 个交易日前的值'),
    'Diff': (2, {1: 1}, 'Diff(x, t)：x - Delay(x, t)'),
    'Pct': (2, {1: 1}, 'Pct(x, t)：x 相对 t 日前的变化率'),
    'Ma': (2, {1: 1}, 'Ma(x, t)：过去 t 日均值'),
    'Med': (2, {1: 1}, 'Med(x, t)：过去 t 日中位数'),
    'Sum': (2, {1: 1}, 'Sum(x, t)：过去 t 日求和'),
    'Std': (2, {1: 2}, 'Std(x, t)：过去 t 日标准差'),
    'Max': (2, {1: 1}, 'Max(x, t)：过去 t 日最大值'),
    'Min': (2, {1: 1}, 'Min(x, t)：过去 t 日最小值'),
    'Rank': (2, {1: 2}, 'Rank(x, t)：x 在自身过去 t 日取值中的排名（时间序列排名）'),
    'Skew': (2, {1: 3}, 'Skew(x, t)：过去 t 日偏度'),
    'Kurt': (2, {1: 4}, 'Kurt(x, t)：过去 t 日峰度'),
    'Vari': (2, {1: 2}, 'Vari(x, t)：Std(x, t) / Ma(x, t)'),
    'Autocorr': (3, {1: 2, 2: 1}, 'Autocorr(x, t, n)：过去 t 日内 x 与其 n 日滞后值的自相关系数'),
    'Zscore': (2, {1: 2}, 'Zscore(x, t)：基于过去 t 日均值和标准差的标准化'),
    'Greater': (2, {}, 'Greater(x, y)：x 是否大于 y（1/0）'),
    'Less': (2, {}, 'Less(x, y)：x 是否小于 y（1/0）'),
    'Cov': (3, {2: 2}, 'Cov(x, y, t)：过去 t 日协方差'),
    'Corr': (3, {2: 2}, 'Corr(x, y, t)：过去 t 日 Pearson 相关系数'),
}
MAX_WINDOW = 250
MAX_PARAMETERS = 3
_BINARY = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}


class FormulaError(ValueError):
    """An invalid formula; the message is written for the LLM correction step."""


def fields_text(fields):
    return ', '.join('$' + f for f in fields)


def operators_text():
    return '\n'.join(f'- {desc}' if '(' in desc else f'- {name}(x)：{desc}'
                     for name, (_, _, desc) in OPERATORS.items()) + \
        '\n- 四则运算 + - * /，取负 -x，平方 x**2，倒数 1/x' + \
        '\n注意：所有算子都在单只股票的时间序列上计算（Rank 也是时间序列排名，必须给出窗口 t）；' \
        '评估时会按日对因子做横截面比较，公式中不需要也无法写横截面排名。'


def parse(expr):
    if not isinstance(expr, str) or not expr.strip():
        raise FormulaError('公式为空')
    text = expr.strip().strip('`').strip()
    if re.search(r'\$\$|\$\s', text):
        raise FormulaError('字段写法错误：字段应写成 $close 这样的形式')
    try:
        return ast.parse(text.replace('$', 'field_'), mode='eval')
    except SyntaxError as error:
        raise FormulaError(f'表达式语法错误（{error.msg}），请检查括号与逗号') from None


def _number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number(node.operand)
        return None if value is None else -value
    return None


def check(expr, fields=ALL_FIELDS, parameters=None):
    """Raise FormulaError unless ``expr`` is a well-formed formula.

    ``parameters`` maps symbolic parameter names to concrete values. Names that
    are neither fields nor operators are treated as symbolic parameters; at most
    three are allowed, and window arguments must resolve to positive integers.
    Returns the set of parameter names used.
    """
    tree = parse(expr)
    parameters = parameters or {}
    names, used_fields = set(), set()

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise FormulaError(f'不支持的常量：{node.value!r}')
            return
        if isinstance(node, ast.Name):
            if node.id.startswith('field_'):
                field = node.id[len('field_'):]
                if field not in fields:
                    raise FormulaError(f'字段 ${field} 不可用；只能使用 {fields_text(fields)}')
                used_fields.add(field)
            elif node.id in OPERATORS:
                raise FormulaError(f'算子 {node.id} 必须以函数形式调用')
            else:
                names.add(node.id)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            visit(node.left)
            visit(node.right)
            if isinstance(node.op, ast.Pow) and _number(node.right) is None \
                    and not isinstance(node.right, ast.Name):
                raise FormulaError('幂运算的指数必须是数值')
            return
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name not in OPERATORS:
                raise FormulaError(f'未知算子 {name or ast.unparse(node.func)}；只能使用：'
                                   + ', '.join(OPERATORS))
            arity, windows, _ = OPERATORS[name]
            if node.keywords or len(node.args) != arity:
                raise FormulaError(f'{name} 需要 {arity} 个参数，实际为 {len(node.args)} 个')
            for index, arg in enumerate(node.args):
                if index in windows:
                    _check_window(name, arg, windows[index], parameters, names)
                else:
                    visit(arg)
            return
        if isinstance(node, ast.Compare):
            raise FormulaError('不支持比较运算符；请使用 Greater(x, y) 或 Less(x, y)')
        raise FormulaError(f'不支持的表达式结构：{type(node).__name__}')

    visit(tree)
    if not used_fields:
        raise FormulaError('公式没有引用任何数据字段')
    if len(names) > MAX_PARAMETERS:
        raise FormulaError(f'参数数量 {len(names)} 超过上限 {MAX_PARAMETERS}：{sorted(names)}')
    missing = [n for n in names if parameters and n not in parameters]
    if missing:
        raise FormulaError(f'参数 {missing} 没有给出取值')
    return names


def _check_window(op, node, minimum, parameters, names):
    if isinstance(node, ast.Name) and not node.id.startswith('field_') and node.id not in OPERATORS:
        names.add(node.id)
        if node.id not in parameters:
            return
        value = parameters[node.id]
    else:
        value = _number(node)
        if value is None:
            raise FormulaError(f'{op} 的窗口参数必须是整数或具名参数，而不是表达式 {ast.unparse(node)}')
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FormulaError(f'{op} 的窗口必须是整数，实际为 {value!r}')
    if value < minimum or value > MAX_WINDOW:
        raise FormulaError(f'{op} 的窗口必须在 [{minimum}, {MAX_WINDOW}] 内，实际为 {value}'
                           + ('；禁止使用未来数据' if value < 0 else ''))


def validate(expr, fields=ALL_FIELDS, parameters=None):
    try:
        check(expr, fields, parameters)
        return True, None
    except FormulaError as error:
        return False, str(error)


def _format_number(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return repr(value) if isinstance(value, float) else str(value)


def render(node, abstract=False, parameters=None):
    """Render a parsed formula in the paper syntax.

    ``abstract`` replaces every number and symbolic parameter with ``t`` so that
    formulas differing only in parameters share one structure.
    """
    parameters = parameters or {}
    if isinstance(node, str):
        node = parse(node)
    if isinstance(node, ast.Expression):
        node = node.body

    def go(n):
        if isinstance(n, ast.Constant):
            return 't' if abstract else _format_number(n.value)
        if isinstance(n, ast.Name):
            if n.id.startswith('field_'):
                return '$' + n.id[len('field_'):]
            if abstract:
                return 't'
            return _format_number(parameters[n.id]) if n.id in parameters else n.id
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                value = _number(n)
                if value is not None and not abstract:
                    return _format_number(value)
                return f'(-{go(n.operand)})'
            return go(n.operand)
        if isinstance(n, ast.BinOp):
            return f'({go(n.left)} {_BINARY[type(n.op)]} {go(n.right)})'
        if isinstance(n, ast.Call):
            return f'{n.func.id}({", ".join(go(a) for a in n.args)})'
        raise FormulaError(f'不支持的表达式结构：{type(n).__name__}')

    return go(node)


def substitute(expr, parameters):
    return render(parse(expr), parameters=parameters)


def structure(expr):
    """Parameter-free structural key of a formula."""
    return render(parse(expr), abstract=True)


def to_qlib(expr):
    """Translate a concrete paper-syntax formula into a Qlib expression."""
    tree = parse(expr)

    def go(n):
        if isinstance(n, ast.Expression):
            return go(n.body)
        if isinstance(n, ast.Constant):
            return _format_number(n.value)
        if isinstance(n, ast.Name):
            if not n.id.startswith('field_'):
                raise FormulaError(f'参数 {n.id} 尚未替换为具体数值')
            return '$' + n.id[len('field_'):]
        if isinstance(n, ast.UnaryOp):
            value = _number(n)
            if value is not None:
                return _format_number(value)
            return f'(0 - {go(n.operand)})' if isinstance(n.op, ast.USub) else go(n.operand)
        if isinstance(n, ast.BinOp):
            left, right = go(n.left), go(n.right)
            if isinstance(n.op, ast.Pow):
                return f'Power({left}, {right})'
            return f'({left} {_BINARY[type(n.op)]} {right})'
        name, args = n.func.id, [go(a) for a in n.args]
        if name == 'Delay':
            return f'Ref({args[0]}, {args[1]})'
        if name == 'Diff':
            return f'Delta({args[0]}, {args[1]})'
        if name == 'Pct':
            return f'({args[0]} / Ref({args[0]}, {args[1]}) - 1)'
        if name == 'Ma':
            return f'Mean({args[0]}, {args[1]})'
        if name == 'Vari':
            return f'(Std({args[0]}, {args[1]}) / Mean({args[0]}, {args[1]}))'
        if name == 'Zscore':
            return f'(({args[0]} - Mean({args[0]}, {args[1]})) / Std({args[0]}, {args[1]}))'
        if name == 'Autocorr':
            return f'Corr({args[0]}, Ref({args[0]}, {args[2]}), {args[1]})'
        if name == 'Greater':
            return f'Gt({args[0]}, {args[1]})'
        if name == 'Less':
            return f'Lt({args[0]}, {args[1]})'
        return f'{name}({", ".join(args)})'

    return go(tree)


def is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
