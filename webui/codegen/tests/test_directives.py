"""Unit tests for the #if/#elif/#else/#endif resolver — the codegen core.

Standalone (no pytest needed, matching the repo convention):
    python webui/codegen/tests/test_directives.py
Also works under pytest if available.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # webui/

from codegen.directives import resolve, evaluate, DirectiveError   # noqa: E402


@contextlib.contextmanager
def raises(exc):
    try:
        yield
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


# ── basic keep/drop ──────────────────────────────────────────────────
def test_simple_true_keeps_then_branch():
    src = "a\n#if X\nb\n#else\nc\n#endif\nd\n"
    assert resolve(src, {"X": 1}) == "a\nb\nd\n"

def test_simple_false_keeps_else_branch():
    src = "a\n#if X\nb\n#else\nc\n#endif\nd\n"
    assert resolve(src, {"X": 0}) == "a\nc\nd\n"

def test_no_directives_is_identity():
    src = "line1\n  line2\n\tline3\nno-newline-at-eof"
    assert resolve(src, {}) == src

def test_live_lines_are_byte_identical():
    src = "#if X\n    keep me\n\n        and me\n#endif\n"
    assert resolve(src, {"X": 1}) == "    keep me\n\n        and me\n"


# ── multi-way: exactly one branch (first match wins) ─────────────────
def test_elif_first_true_wins_even_if_later_also_true():
    src = "#if M==1\nA\n#elif M==2\nB\n#elif M>=0\nC\n#else\nD\n#endif\n"
    assert resolve(src, {"M": 2}) == "B\n"
    assert resolve(src, {"M": 1}) == "A\n"
    assert resolve(src, {"M": 5}) == "C\n"
    assert resolve(src, {"M": -1}) == "D\n"


# ── nesting: parent gating ───────────────────────────────────────────
def test_nested_true_inside_dead_parent_is_dropped():
    src = "#if OUTER\n#if INNER\nX\n#else\nY\n#endif\n#else\nZ\n#endif\n"
    assert resolve(src, {"OUTER": 0, "INNER": 1}) == "Z\n"
    assert resolve(src, {"OUTER": 1, "INNER": 0}) == "Y\n"
    assert resolve(src, {"OUTER": 1, "INNER": 1}) == "X\n"

def test_dead_branch_skips_evaluation_of_unknown_symbol():
    src = "#if X\nkeep\n#else\n#if UNKNOWN\nq\n#endif\n#endif\n"
    assert resolve(src, {"X": 1}) == "keep\n"


# ── directive vs content recognition ─────────────────────────────────
def test_pragma_define_include_pass_through():
    src = "#if X\n#pragma unroll\n#define FOO 1\n#include <c>\nbody\n#endif\n"
    assert resolve(src, {"X": 1}) == "#pragma unroll\n#define FOO 1\n#include <c>\nbody\n"

def test_comment_mentioning_if_is_not_a_directive():
    src = "#if X\n// use #if for conditionals\n#endif\n"
    assert resolve(src, {"X": 1}) == "// use #if for conditionals\n"

def test_indented_directives_recognized():
    src = "#if X\n    #if Y\n        deep\n    #endif\n#endif\n"
    assert resolve(src, {"X": 1, "Y": 1}) == "        deep\n"


# ── expression evaluator ─────────────────────────────────────────────
def test_eval_forms():
    d = {"A": 8, "B": 0}
    assert evaluate("A == 8", d) == 1
    assert evaluate("A != 8", d) == 0
    assert evaluate("A > 4 and not B", d) == 1
    assert evaluate("B or A >= 8", d) == 1
    assert evaluate("A + 1 == 9", d) == 1


# ── error cases ──────────────────────────────────────────────────────
def test_unknown_symbol_in_live_branch_raises():
    with raises(DirectiveError):
        resolve("#if NOPE\nx\n#endif\n", {})

def test_unbalanced_endif_raises():
    with raises(DirectiveError):
        resolve("x\n#endif\n", {})

def test_unterminated_if_raises():
    with raises(DirectiveError):
        resolve("#if X\nx\n", {"X": 1})

def test_double_else_raises():
    with raises(DirectiveError):
        resolve("#if X\na\n#else\nb\n#else\nc\n#endif\n", {"X": 1})

def test_elif_after_else_raises():
    with raises(DirectiveError):
        resolve("#if X\na\n#else\nb\n#elif Y\nc\n#endif\n", {"X": 0, "Y": 1})

def test_if_without_expression_raises():
    with raises(DirectiveError):
        resolve("#if\nx\n#endif\n", {})


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
