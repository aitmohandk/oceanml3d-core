"""The documentation stays true to the repository.

Two things drift silently in a documented project: links (a file moves, a heading is renamed, and the
reference rots) and completeness (a model, a task or an ablation is added and the entry point never
mentions it). The README is meant to list *everything* the repository ships; these tests make that a
property rather than an intention.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAINTAINED = sorted([REPO / "README.md", REPO / "jobs" / "README.md",
                     *(p for p in (REPO / "docs").rglob("*.md") if "legacy" not in p.parts)])
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, markup and punctuation dropped, spaces to '-'."""
    text = re.sub(r"`|\*|_(?=\w)|(?<=\w)_", "", heading.strip()).lower()
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if c.isalnum() or c in " -_")
    return text.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out, in_code = set(), False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code and line.startswith("#"):
            out.add(github_slug(line.lstrip("#")))
    return out


def _links(path: Path):
    in_code = False
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for target in LINK.findall(line):
            if not re.match(r"[a-z]+://|mailto:", target):
                yield n, target


@pytest.mark.parametrize("doc", MAINTAINED, ids=lambda p: str(p.relative_to(REPO)))
def test_every_relative_link_and_anchor_resolves(doc):
    broken = []
    for n, target in _links(doc):
        file_part, _, anchor = target.partition("#")
        dest = (doc.parent / file_part).resolve() if file_part else doc
        if not dest.exists():
            broken.append(f"line {n}: {target} -> {dest} does not exist")
        elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            broken.append(f"line {n}: {target} -> no heading '#{anchor}' in {dest.name}")
    assert not broken, "\n".join(broken)


def _names(group: str) -> set[str]:
    return {p.stem for p in (REPO / "config" / group).glob("*.yaml")}


@pytest.mark.parametrize("group", ["data", "experiment", "model", "ablation", "paths"])
def test_the_readme_names_everything_the_repository_ships(group):
    """`README.md` §3 is the complete list. Adding a config without a line there fails here."""
    readme = (REPO / "README.md").read_text()
    missing = sorted(n for n in _names(group) if f"`{n}`" not in readme)
    assert not missing, f"config/{group}/: {missing} not mentioned in README.md"


def test_the_readme_lists_every_cli_command():
    commands = set(re.findall(r'cmd == "([a-z-]+)"', (REPO / "oceanml3d" / "cli.py").read_text()))
    readme = (REPO / "README.md").read_text()
    missing = sorted(c for c in commands if f"`{c}`" not in readme)
    assert commands and not missing, f"CLI commands not in README.md: {missing}"


def test_the_readme_lists_every_preparation_step():
    steps = set(re.findall(r"^\s+([a-z]+)\)\s+step_", (REPO / "jobs" / "prepare.sh").read_text(), re.M))
    readme = (REPO / "README.md").read_text()
    missing = sorted(s for s in steps if f"`{s}" not in readme)
    assert steps and not missing, f"jobs/prepare.sh steps not in README.md: {missing}"


def test_the_slug_rule_matches_github_on_the_headings_this_repository_uses():
    assert github_slug("3. What it does — the complete list") == "3-what-it-does--the-complete-list"
    assert github_slug("H. Inference, and getting results off `$SCRATCH`") == \
        "h-inference-and-getting-results-off-scratch"
    assert github_slug("3.4 Ablations — `config/ablation/`") == "34-ablations--configablation"
    assert github_slug("Part 1 — A rehearsal on your machine") == "part-1--a-rehearsal-on-your-machine"
