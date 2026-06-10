"""Resolve ``#if`` / ``#elif`` / ``#else`` / ``#endif`` against known integer
values — a strict, line-oriented subset of the C preprocessor's *conditional
inclusion*.

This is the core of mmcomposer's "generate a kernel specialized to one knob
combo" step: every knob value is known at generation time, so every directive
resolves to exactly one branch and nothing is left conditional.

References for the semantics + a real-world implementation of the same idea
(selectively resolving ``#if`` for known symbols):
  * cppreference, Conditional inclusion: https://en.cppreference.com/c/preprocessor/conditional
  * GCC CPP manual, Conditionals: https://gcc.gnu.org/onlinedocs/cpp/Conditionals.html
  * unifdef (Tony Finch): https://dotat.at/prog/unifdef/  (src: github.com/fanf2/unifdef)

Design contract:
  * Single top-to-bottom pass; nesting tracked on a stack (one frame per open
    ``#if`` chain).  See `resolve` for the per-line transitions.
  * **Content lines are never rewritten — only kept or dropped.** Directive
    lines are always dropped.  (Constant substitution is a separate step.)
  * Conditions are evaluated only where they decide output (live regions); a
    ``#if`` buried in an already-dead branch is skipped, so an unknown symbol
    there can't raise — matching the C preprocessor.
"""

from __future__ import annotations

import ast
import re

__all__ = ["resolve", "evaluate", "DirectiveError"]

# A directive line is one whose first non-space char is '#' followed by one of
# our four keywords.  This deliberately does NOT match #define / #include /
# #pragma (they pass through as ordinary content), nor comment lines (which
# start with '//', not '#').
_DIRECTIVE = re.compile(r"#\s*(if|elif|else|endif)\b(.*)$", re.DOTALL)

_CMP = {
    ast.Eq: lambda a, b: a == b,   ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,    ast.GtE: lambda a, b: a >= b,
}


class DirectiveError(ValueError):
    """Malformed directive, unknown symbol, or unbalanced #if/#endif."""


def evaluate(expr: str, defines: dict) -> int:
    """Evaluate a ``#if``/``#elif`` integer-constant expression over `defines`.

    Supports: integer literals, knob names (looked up in `defines`; unknown
    name -> DirectiveError), comparisons (== != < > <= >=), boolean and/or/not,
    and + - * .  Uses a restricted AST walk — never Python ``eval``.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise DirectiveError(f"cannot parse directive expression {expr!r}: {e}") from None
    return _eval_node(tree.body, defines, expr)


def _eval_node(node, defines, expr):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise DirectiveError(f"non-integer constant in directive {expr!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in defines:
            raise DirectiveError(f"unknown symbol {node.id!r} in directive {expr!r}")
        return int(defines[node.id])
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, defines, expr) for v in node.values]
        return int(all(vals) if isinstance(node.op, ast.And) else any(vals))
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, defines, expr)
        if isinstance(node.op, ast.Not):  return int(not v)
        if isinstance(node.op, ast.USub): return -v
        if isinstance(node.op, ast.UAdd): return +v
    if isinstance(node, ast.BinOp):
        a = _eval_node(node.left, defines, expr)
        b = _eval_node(node.right, defines, expr)
        if isinstance(node.op, ast.Add):  return a + b
        if isinstance(node.op, ast.Sub):  return a - b
        if isinstance(node.op, ast.Mult): return a * b
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, defines, expr)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, defines, expr)
            fn = _CMP.get(type(op))
            if fn is None:
                raise DirectiveError(f"unsupported comparison in directive {expr!r}")
            if not fn(left, right):
                return 0
            left = right
        return 1
    raise DirectiveError(f"unsupported expression element in directive {expr!r}")


class _Frame:
    """One open ``#if`` chain.  `active` is already ANDed with `parent_emitting`,
    so 'are we emitting?' is just the top frame's `active`."""
    __slots__ = ("parent_emitting", "active", "taken", "seen_else")

    def __init__(self, parent_emitting: bool, active: bool, taken: bool):
        self.parent_emitting = parent_emitting
        self.active = active
        self.taken = taken
        self.seen_else = False


def resolve(src: str, defines: dict) -> str:
    """Return `src` with every knob conditional resolved against `defines`:
    dead branches and all directive lines removed; live lines kept verbatim."""
    out: list[str] = []
    stack: list[_Frame] = []
    for lineno, line in enumerate(src.splitlines(keepends=True), 1):
        m = _DIRECTIVE.match(line.strip())
        if not m:
            if (stack[-1].active if stack else True):   # emitting?
                out.append(line)
            continue
        kw, rest = m.group(1), m.group(2).strip()
        if kw in ("if", "elif") and not rest:
            raise DirectiveError(f"line {lineno}: #{kw} needs an expression")

        if kw == "if":
            parent = stack[-1].active if stack else True
            cond = bool(evaluate(rest, defines)) if parent else False
            stack.append(_Frame(parent_emitting=parent, active=cond, taken=cond))
        elif kw == "elif":
            if not stack:
                raise DirectiveError(f"line {lineno}: #elif without #if")
            f = stack[-1]
            if f.seen_else:
                raise DirectiveError(f"line {lineno}: #elif after #else")
            if f.parent_emitting and not f.taken:
                c = bool(evaluate(rest, defines))   # only evaluated where it decides output
                f.active = c
                f.taken = f.taken or c
            else:
                f.active = False
        elif kw == "else":
            if not stack:
                raise DirectiveError(f"line {lineno}: #else without #if")
            f = stack[-1]
            if f.seen_else:
                raise DirectiveError(f"line {lineno}: duplicate #else")
            f.seen_else = True
            f.active = f.parent_emitting and not f.taken
            f.taken = True
        else:  # endif
            if not stack:
                raise DirectiveError(f"line {lineno}: #endif without #if")
            stack.pop()

    if stack:
        raise DirectiveError("unterminated #if (missing #endif)")
    return "".join(out)
