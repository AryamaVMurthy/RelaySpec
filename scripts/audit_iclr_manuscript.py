#!/usr/bin/env python3
"""Audit the RelaySpec ICLR manuscript for format, evidence, and style."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OFFICIAL_HASHES = {
    "iclr2027_conference.sty": "797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6",
    "iclr2027_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
}
REQUIRED_SECTIONS = (
    "Introduction",
    "Related work and positioning",
    "Preliminaries",
    "Method",
    "Experiments",
    "Conclusion",
)

BANNED_PHRASES = (
    "first-error localization",
    "transfer dynamics",
    "representation quality",
    "specialization-generalization trade-off",
    "state of the art",
    "universal",
    "lossless",
    "seamless",
)
IDENTITY_MARKERS = (
    "aryamavmurthy",
    "aryama.murthy",
    "/home/",
    "/scratch/",
    "iiit.ac.in",
    "ideaPad",
)
STATUS_PHRASES = (
    "Under review as a conference paper",
    "Published as a conference paper",
)
ENGINEERING_DETAIL_PATTERNS = (
    r"\b[0-9a-f]{8,40}\b",
    r"\bmanifest digest\b",
    r"\ballocation proof\b",
    r"\bGPU telemetry\b",
    r"\bpassed all \d+ repository tests\b",
    r"\bseed\s+\d+\b",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _strip_comments(source: str) -> str:
    lines: list[str] = []
    for line in source.splitlines():
        cut = len(line)
        for match in re.finditer(r"%", line):
            index = match.start()
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        lines.append(line[:cut])
    return "\n".join(lines)


def _strip_drawing_environments(source: str) -> str:
    """Remove tikzpicture and algorithmic bodies before checking prose.

    Both environments require a trailing semicolon (TikZ) or use it as
    pseudocode punctuation (algorithmic), which is standard LaTeX/TikZ
    syntax, not prose. The language-constraint check below only cares about
    semicolons and em dashes in sentences a reader sees as English text.
    """
    pattern = re.compile(
        r"\\begin\{(tikzpicture|algorithmic)\}.*?\\end\{\1\}", re.DOTALL
    )
    return pattern.sub("", source)


def _citation_keys(source: str) -> list[str]:
    keys: list[str] = []
    for match in re.finditer(r"\\cite\w*\{([^}]+)\}", source):
        keys.extend(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def _bib_keys(source: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", source))


def _main_page(aux: str) -> int | None:
    match = re.search(r"\\newlabel\{maintextend\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(match.group(1)) if match else None


def _pdf_info(path: Path) -> dict[str, str]:
    process = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[str, str] = {}
    for line in process.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _pdf_text(path: Path) -> str:
    process = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout


def _normalized_paragraphs(source: str) -> list[str]:
    paragraphs: list[str] = []
    for raw in re.split(r"\n\s*\n", source):
        if raw.lstrip().startswith("\\"):
            continue
        text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", raw)
        text = re.sub(r"[{}$^_\\]", " ", text)
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().strip()
        if len(text.split()) >= 35:
            paragraphs.append(text)
    return paragraphs


def _paragraph_overlap(
    source: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
    paragraphs = _normalized_paragraphs(source)
    exact: list[tuple[int, int]] = []
    similar: list[tuple[int, int, float]] = []
    for left in range(len(paragraphs)):
        for right in range(left + 1, len(paragraphs)):
            if paragraphs[left] == paragraphs[right]:
                exact.append((left, right))
                continue
            left_tokens = set(paragraphs[left].split())
            right_tokens = set(paragraphs[right].split())
            score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if score >= 0.90:
                similar.append((left, right, score))
    return exact, similar


def _generated_assets_match(root: Path, paper: Path) -> tuple[bool, str]:
    builder_path = root / "scripts" / "build_iclr_paper_assets.py"
    spec = importlib.util.spec_from_file_location(
        "paper_assets_for_audit", builder_path
    )
    if spec is None or spec.loader is None:
        return False, "asset builder could not be imported"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="relayspec-paper-audit-") as directory:
        expected = Path(directory)
        module.build_all(root, expected)
        relative_files = sorted(
            path.relative_to(expected) for path in expected.rglob("*") if path.is_file()
        )
        for relative in relative_files:
            actual = paper / relative
            if not actual.exists():
                return False, f"missing generated asset {relative}"
            if actual.read_bytes() != (expected / relative).read_bytes():
                return False, f"stale generated asset {relative}"
    for registry_name, builder_name in [
        ("target14_evidence_registry.json", "build_target14_paper_assets.py"),
        ("scaling_evidence_registry.json", "build_scaling_paper_assets.py"),
        ("capacity_evidence_registry.json", "build_capacity_paper_assets.py"),
        (
            "regularization_evidence_registry.json",
            "build_regularization_paper_assets.py",
        ),
        (
            "eagle_capacity_evidence_registry.json",
            "build_eagle_capacity_paper_assets.py",
        ),
        ("small_quality_evidence_registry.json", "build_small_quality_paper_assets.py"),
        ("eagle_quality_evidence_registry.json", "build_eagle_quality_paper_assets.py"),
        ("resumed_evidence_registry.json", "build_resumed_paper_assets.py"),
        (
            "task_complexity_evidence_registry.json",
            "build_task_complexity_paper_assets.py",
        ),
        ("checkpoint_evidence_registry.json", "build_checkpoint_paper_assets.py"),
        ("timed_budget_evidence_registry.json", "build_timed_budget_paper_assets.py"),
        (
            "external_baseline_evidence_registry.json",
            "build_external_baseline_paper_assets.py",
        ),
    ]:
        registry = paper / "generated" / registry_name
        if not registry.exists():
            continue
        raw_root = json.loads(registry.read_text())["raw_root"]
        with tempfile.TemporaryDirectory(
            prefix="relayspec-scaling-audit-"
        ) as directory:
            expected = Path(directory)
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / builder_name),
                    "--raw-root",
                    raw_root,
                    "--output",
                    str(expected),
                ],
                cwd=root,
                text=True,
                capture_output=True,
            )
            if result.returncode:
                return (
                    False,
                    "generated evidence source validation failed: "
                    + result.stderr[-1000:],
                )
            for path in expected.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(expected)
                    if (
                        not (paper / relative).exists()
                        or (paper / relative).read_bytes() != path.read_bytes()
                    ):
                        return False, f"stale evidence asset {relative}"
    research = _load_script(
        root / "scripts/audit_autoresearch_assets.py", "autoresearch_assets_for_audit"
    )
    research_pass, research_evidence = research.audit(root)
    if not research_pass:
        return False, research_evidence
    return (
        True,
        "registered core assets match validated evidence; " + research_evidence,
    )


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompt_separation_matches(root: Path) -> tuple[bool, str]:
    fit = json.loads(
        (root / "configs" / "train_math_4096.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (root / "configs" / "eval_manifest_full_v4.json").read_text(encoding="utf-8")
    )
    exact_module = _load_script(
        root / "scripts" / "check_prompt_overlap.py", "paper_exact_overlap_audit"
    )
    similarity_module = _load_script(
        root / "scripts" / "audit_prompt_similarity.py",
        "paper_prompt_similarity_audit",
    )
    exact = exact_module.audit_prompt_overlap(fit, evaluation)
    similarity = similarity_module.audit_prompt_similarity(fit, evaluation)
    stored_exact = json.loads(
        (root / "reports" / "final" / "PROMPT_OVERLAP_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    stored_similarity = json.loads(
        (root / "reports" / "final" / "PROMPT_SIMILARITY_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    passed = exact == stored_exact and similarity == stored_similarity
    passed = passed and exact["overlap_count"] == 0
    return (
        passed,
        "exact-overlap and unique-token similarity artifacts reproduce from the fixed manifests",
    )


def _pdf_fonts_and_parser(path: Path) -> tuple[bool, str]:
    font_process = subprocess.run(
        ["pdffonts", str(path)], check=True, capture_output=True, text=True
    )
    font_rows = font_process.stdout.splitlines()[2:]
    embedding = [
        re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row)
        for row in font_rows
        if row.strip()
    ]
    fonts_embedded = bool(embedding) and all(
        match is not None and match.group(1) == "yes" for match in embedding
    )
    parser = subprocess.run(
        [
            "gs",
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=nullpage",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        fonts_embedded and parser.returncode == 0,
        f"{len(embedding)} font records are embedded and Ghostscript parses every page",
    )


def _recorded_visual_review(root: Path, pdf: Path, pages: int) -> tuple[bool, str]:
    path = root / "reports" / "ICLR_VISUAL_REVIEW.json"
    if not path.exists():
        return False, "manual visual review record is missing"
    review = json.loads(path.read_text(encoding="utf-8"))
    required_true = (
        "page_by_page_review",
        "no_clipping_or_overlap",
        "tables_and_equations_legible",
        "figures_legible_in_color",
        "figures_legible_in_grayscale",
        "float_order_reviewed",
        "link_boxes_hidden",
    )
    passed = review.get("pdf_sha256") == _sha256(pdf)
    passed = passed and review.get("pages") == pages
    passed = passed and review.get("color_rendered_pages") == pages
    passed = passed and review.get("grayscale_rendered_pages") == pages
    passed = passed and all(review.get(key) is True for key in required_true)
    if passed:
        return (
            True,
            f"manual color and grayscale review covers all {pages} rendered pages",
        )
    return False, (
        f"current {pages}-page PDF lacks a complete matching visual review "
        f"(recorded {review.get('pages')} pages, matching PDF hash: "
        f"{review.get('pdf_sha256') == _sha256(pdf)})"
    )


def _check(name: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def audit_manuscript(root: Path, *, write_report: bool = True) -> dict[str, Any]:
    root = root.resolve()
    paper = root / "paper" / "iclr2027"
    tex_path = paper / "relayspec_iclr2027.tex"
    bib_path = paper / "references.bib"
    aux_path = paper / "relayspec_iclr2027.aux"
    log_path = paper / "relayspec_iclr2027.log"
    pdf_path = paper / "relayspec_iclr2027.pdf"

    source_raw = tex_path.read_text(encoding="utf-8")
    source = _strip_comments(source_raw)
    bib = bib_path.read_text(encoding="utf-8")
    aux = aux_path.read_text(encoding="utf-8")
    log = log_path.read_text(encoding="utf-8")
    pdf_info = _pdf_info(pdf_path)
    pdf_text = _pdf_text(pdf_path)
    citation_keys = _citation_keys(source)
    missing_citations = sorted(set(citation_keys) - _bib_keys(bib))
    main_page = _main_page(aux)
    exact_paragraphs, similar_paragraphs = _paragraph_overlap(source)
    generated_pass, generated_evidence = _generated_assets_match(root, paper)
    separation_pass, separation_evidence = _prompt_separation_matches(root)

    style_ok = "\\usepackage{iclr2027_conference,times}" in source
    style_ok = style_ok and all(
        (paper / filename).exists() and _sha256(paper / filename) == digest
        for filename, digest in OFFICIAL_HASHES.items()
    )
    active_final = bool(re.search(r"\\iclrfinalcopy\b", source))
    identity_hits = sorted(
        marker
        for marker in IDENTITY_MARKERS
        if marker.lower() in source.lower() or marker.lower() in pdf_text.lower()
    )
    anonymous_ok = (
        not active_final and not identity_hits and "Anonymous authors" in pdf_text
    )
    status_hits = [
        phrase for phrase in STATUS_PHRASES if phrase.lower() in pdf_text.lower()
    ]
    clean_header_ok = (
        "Under review as a conference paper at ICLR 2027" in pdf_text
        and "Published as a conference paper at ICLR 2027" not in pdf_text
        and "\\lhead" not in source
        and "\\fancyhead" not in source
    )
    engineering_detail_hits = [
        pattern
        for pattern in ENGINEERING_DETAIL_PATTERNS
        if re.search(
            pattern, source.split("\\label{maintextend}")[0], flags=re.IGNORECASE
        )
    ]

    required_missing = [
        section
        for section in REQUIRED_SECTIONS
        if f"\\section{{{section}}}" not in source
    ]
    statements_ok = all(
        heading in source
        for heading in (
            "\\section*{AI use statement}",
            "\\section*{Reproducibility statement}",
            "\\section*{Appendix}",
        )
    )
    document_order_ok = (
        source.index("\\label{maintextend}")
        < source.index("\\section*{AI use statement}")
        < source.index("\\bibliography{references}")
        < source.index("\\appendix")
    )
    language_hits: list[str] = []
    prose_source = _strip_drawing_environments(source)
    if ";" in prose_source:
        language_hits.append("semicolon")
    if "—" in prose_source or "---" in prose_source:
        language_hits.append("em dash")
    for phrase in BANNED_PHRASES:
        if phrase.lower() in prose_source.lower():
            language_hits.append(phrase)
    placeholder_hits = re.findall(
        r"\b(?:TODO|TBD|XX|citation needed)\b|\?\?", source, flags=re.IGNORECASE
    )

    overlap_artifact = json.loads(
        (root / "reports" / "final" / "PROMPT_OVERLAP_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    log_ok = not re.search(
        r"undefined|Overfull \\hbox|Overfull \\vbox", log, re.IGNORECASE
    )
    pdf_ok = (
        pdf_info.get("Page size", "").startswith("612 x 792")
        and pdf_info.get("Encrypted") == "no"
        and not any(marker.lower() in pdf_text.lower() for marker in IDENTITY_MARKERS)
    )
    fonts_pass, fonts_evidence = _pdf_fonts_and_parser(pdf_path)
    visual_pass, visual_evidence = _recorded_visual_review(
        root, pdf_path, int(pdf_info.get("Pages", "0"))
    )

    checks = [
        _check(
            "official ICLR 2027 style",
            style_ok,
            "official style and bibliography files match recorded hashes",
        ),
        _check(
            "anonymous review source",
            anonymous_ok,
            "review mode is active and no local identity marker appears in source or PDF",
        ),
        _check(
            "clean anonymous status header",
            clean_header_ok,
            "the official anonymous review header is present without author overrides",
        ),
        _check(
            "reader-facing scientific detail",
            not engineering_detail_hits,
            "main prose omits internal engineering records, while appendix retains reproducibility settings",
        ),
        _check(
            "nine-page main-text limit",
            main_page is not None and main_page <= 9,
            f"main-text boundary is on page {main_page}",
        ),
        _check(
            "required sections and order",
            not required_missing and statements_ok and document_order_ok,
            "all main sections and policy statements are present before references and appendix",
        ),
        _check(
            "citation resolution",
            not missing_citations
            and len(set(citation_keys)) >= 42
            and "undefined citations" not in log.lower(),
            f"{len(set(citation_keys))} unique citation keys resolve",
        ),
        _check("generated result assets", generated_pass, generated_evidence),
        _check("prompt separation audits", separation_pass, separation_evidence),
        _check(
            "language constraints",
            not language_hits and not placeholder_hits,
            "no semicolon, em dash, banned phrase, or placeholder appears in manuscript source",
        ),
        _check(
            "paragraph overlap",
            not exact_paragraphs and not similar_paragraphs,
            "no exact or at least 0.90 Jaccard duplicate among paragraphs of 35 or more words",
        ),
        _check(
            "fit and evaluation exact-overlap audit",
            overlap_artifact.get("overlap_count") == 0
            and overlap_artifact.get("fit_records") == 4096
            and overlap_artifact.get("evaluation_records") == 1250,
            "0 normalized exact matches across 4,096 fit and 1,250 evaluation records",
        ),
        _check(
            "compiled LaTeX log",
            log_ok,
            "no undefined citation, undefined reference, or overfull box",
        ),
        _check(
            "PDF parse and page format",
            pdf_ok,
            f"{pdf_info.get('Pages')} pages, US Letter, unencrypted, and identity scan clean",
        ),
        _check("embedded fonts and PDF parser", fonts_pass, fonts_evidence),
        _check("recorded visual review", visual_pass, visual_evidence),
    ]

    result = {
        "checks": checks,
        "main_text_end_page": main_page,
        "total_pdf_pages": int(pdf_info.get("Pages", "0")),
        "citation_count": len(set(citation_keys)),
        "missing_citations": missing_citations,
        "identity_hits": identity_hits,
        "status_hits": status_hits,
        "engineering_detail_hits": engineering_detail_hits,
        "language_hits": language_hits,
        "placeholder_hits": placeholder_hits,
        "exact_duplicate_paragraphs": exact_paragraphs,
        "similar_paragraphs": similar_paragraphs,
        "all_passed": all(check["passed"] for check in checks),
    }

    if write_report:
        report = root / "reports" / "ICLR_MANUSCRIPT_QA.md"
        lines = [
            "# RelaySpec ICLR 2027 manuscript QA",
            "",
            f"Generated {datetime.now(timezone.utc).date().isoformat()} from the compiled anonymous manuscript and recorded result artifacts.",
            "",
            f"- Overall: **{'PASS' if result['all_passed'] else 'FAIL'}**",
            f"- Main-text boundary: page {main_page} of the allowed 9",
            f"- Complete PDF: {result['total_pdf_pages']} pages including references and appendix",
            f"- Resolved citation keys: {result['citation_count']}",
            "",
            "| Check | Status | Evidence |",
            "|---|---|---|",
        ]
        for check in checks:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(f"| {check['name']} | {status} | {check['evidence']} |")
        lines.extend(
            [
                "",
                "## Scientific review remains separate",
                "",
                "These automated checks cover the implemented artifact, formatting, and consistency conditions. They do not establish experimental correctness or submission readiness.",
                "See reports/ICLR_SUBMISSION_REVIEW_2026-09-05.md for unresolved output agreement, objective equivalence, adaptation cost, checkpoint export, and claim-scope issues.",
                "",
                "## Visual review",
                "",
                "The review record is tied to the compiled PDF SHA-256. It covers every page in color and grayscale. It checks clipping, overlap, table order, equation legibility, figure legibility, and hidden link boxes.",
            ]
        )
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="RelaySpec repository root",
    )
    args = parser.parse_args()
    result = audit_manuscript(args.root)
    print(f"ICLR manuscript audit: {'PASS' if result['all_passed'] else 'FAIL'}")
    for check in result["checks"]:
        print(
            f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['evidence']}"
        )
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
