"""Guard against docstrings that merely restate the test name.

Commit 1e693da1 stamped a ``Checks <test name as words>.`` docstring onto 536 tests in
one pass; the sweep that removed them (SPRINT_docstring-truth-and-double-dedup)
must not be redone by the next well-meaning batch edit. The matcher is the one
that sized that sweep: strip a leading ``checks`` / ``verifies`` / ``ensures`` /
``tests`` / ``asserts``, tokenize the docstring's first line and the test name
the same way, and call the docstring a restatement when it adds no token to the
name and misses at most two of the name's tokens.

This is a heuristic on purpose. A rewritten docstring that reuses the name's
nouns is fine as long as it says something the name does not (``extra_doc`` is
non-empty); a first line built only from the name's words is exactly what the
guard is for. Files are read with ``ast`` -- importing 300 test modules is slow
and has side effects -- following ``test_generation_rollout_boundaries.py``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

_LEADING_VERBS = frozenset(
    {
        "check",
        "checks",
        "verify",
        "verifies",
        "ensure",
        "ensures",
        "test",
        "tests",
        "assert",
        "asserts",
    }
)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "that",
        "with",
        "as",
        "at",
        "by",
        "from",
        "it",
        "its",
        "this",
        "when",
        "into",
        "not",
        "no",
        "via",
        "vs",
        "than",
        "then",
        "if",
        "so",
        "do",
        "does",
        "has",
        "have",
        "their",
        "each",
        "every",
        "all",
        "any",
        "only",
        "still",
        "also",
        "can",
        "will",
        "should",
        "must",
    }
)
_SLACK = 2


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.split(r"[^a-zA-Z0-9]+", text.lower()):
        if not raw or raw in _STOP:
            continue
        token = raw
        if len(token) > 3 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        out.add(token)
    return out


def restates_name(test_name: str, docstring: str) -> bool:
    """The §2.1 matcher: no token beyond the name, at most ``_SLACK`` name tokens missing."""

    first = docstring.strip().splitlines()[0].strip()
    words = first.split()
    if words and words[0].lower().strip(".:,") in _LEADING_VERBS:
        first = " ".join(words[1:])
    doc_tokens = _tokens(first)
    name_tokens = _tokens(test_name.removeprefix("test_"))
    return not (doc_tokens - name_tokens) and len(name_tokens - doc_tokens) <= _SLACK


def _restating_docstrings() -> list[str]:
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            docstring = ast.get_docstring(node, clean=True)
            if docstring and restates_name(node.name, docstring):
                offenders.append(
                    f"{path.relative_to(TESTS_ROOT.parent)}:{node.lineno} {node.name}"
                )
    return offenders


def test_matcher_separates_restatements_from_invariants() -> None:
    """The heuristic's two edges, pinned so a stop-word tweak cannot silently flip it."""

    assert restates_name("test_stats_shape", "Checks stats shape.")
    assert restates_name("test_wan_decode_keeps_layout", "Verifies Wan decode keeps the layout")
    assert not restates_name(
        "test_stats_shape",
        "Queue stats count group slots; version selection belongs to the scheduler.",
    )
    assert not restates_name(
        "test_too_stale", "Checks the knob is refused where it could have no effect."
    )


def test_no_test_docstring_merely_restates_its_name() -> None:
    """Every test docstring must say something its name does not."""

    offenders = _restating_docstrings()
    assert not offenders, (
        "docstrings that only restate the test name (rewrite to the invariant the test pins, "
        "or delete):\n  " + "\n  ".join(offenders)
    )
