"""Phase 5: Cross-domain transfer validation and coverage gap filling."""

from __future__ import annotations

import logging
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from src.common.config import (
    load_settings,
    get_all_primitive_names,
    get_top10_primitives,
    load_domains,
    data_path,
)
from src.common.models import GeneratedQuestion, ExpertResponse
from src.common.storage import read_jsonl_list
from src.generation.phase4_filtering import _FilteredPair

logger = logging.getLogger(__name__)
console = Console()


def build_coverage_matrix(
    pairs: list[_FilteredPair],
) -> dict[str, dict[str, int]]:
    """Build a primitive x domain coverage matrix."""
    matrix = defaultdict(lambda: defaultdict(int))
    for pair in pairs:
        matrix[pair.question.primitive][pair.question.domain] += 1
    return dict(matrix)


def validate_coverage(pairs: list[_FilteredPair]) -> dict:
    """Validate coverage against requirements. Returns a report dict."""
    settings = load_settings()
    val_settings = settings["validation"]

    all_primitives = get_all_primitive_names()
    top10 = get_top10_primitives()
    all_domains = [d["name"] for d in load_domains()]

    matrix = build_coverage_matrix(pairs)

    # Check 1: Every primitive in 4+ domains
    undercovered = []
    for primitive in all_primitives:
        domains_with_data = sum(1 for d in all_domains if matrix.get(primitive, {}).get(d, 0) > 0)
        if domains_with_data < val_settings["min_domains_per_primitive"]:
            undercovered.append(
                {
                    "primitive": primitive,
                    "domains_covered": domains_with_data,
                    "required": val_settings["min_domains_per_primitive"],
                    "missing_domains": [
                        d for d in all_domains if matrix.get(primitive, {}).get(d, 0) == 0
                    ],
                }
            )

    # Check 2: Top-10 primitives have 50+ examples
    top10_gaps = []
    for primitive in top10:
        total = sum(matrix.get(primitive, {}).values())
        if total < val_settings["min_examples_top10_primitives"]:
            top10_gaps.append(
                {
                    "primitive": primitive,
                    "count": total,
                    "required": val_settings["min_examples_top10_primitives"],
                    "deficit": val_settings["min_examples_top10_primitives"] - total,
                }
            )

    # Check 3: No single domain exceeds 25%
    domain_totals = defaultdict(int)
    total_examples = len(pairs)
    for pair in pairs:
        domain_totals[pair.question.domain] += 1

    domain_imbalances = []
    for domain, count in domain_totals.items():
        pct = count / max(total_examples, 1)
        if pct > val_settings["max_single_domain_pct"]:
            domain_imbalances.append(
                {"domain": domain, "count": count, "pct": round(pct, 3)}
            )

    report = {
        "total_examples": total_examples,
        "unique_primitives": len(matrix),
        "undercovered_primitives": undercovered,
        "top10_gaps": top10_gaps,
        "domain_imbalances": domain_imbalances,
        "coverage_matrix": {p: dict(d) for p, d in matrix.items()},
        "passed": len(undercovered) == 0 and len(top10_gaps) == 0 and len(domain_imbalances) == 0,
    }

    return report


def print_coverage_report(report: dict) -> None:
    """Pretty-print the coverage validation report."""
    console.print(f"\n[bold]Coverage Validation Report[/bold]")
    console.print(f"Total examples: {report['total_examples']}")
    console.print(f"Unique primitives: {report['unique_primitives']}")
    console.print(f"Overall: {'[green]PASSED' if report['passed'] else '[red]FAILED'}[/]")

    if report["undercovered_primitives"]:
        console.print(f"\n[yellow]Undercovered primitives ({len(report['undercovered_primitives'])}):[/]")
        for gap in report["undercovered_primitives"]:
            console.print(
                f"  {gap['primitive']}: {gap['domains_covered']}/{gap['required']} domains"
            )

    if report["top10_gaps"]:
        console.print(f"\n[yellow]Top-10 primitive gaps ({len(report['top10_gaps'])}):[/]")
        for gap in report["top10_gaps"]:
            console.print(
                f"  {gap['primitive']}: {gap['count']}/{gap['required']} examples (need {gap['deficit']} more)"
            )

    if report["domain_imbalances"]:
        console.print(f"\n[yellow]Domain imbalances:[/]")
        for imb in report["domain_imbalances"]:
            console.print(f"  {imb['domain']}: {imb['pct']*100:.1f}% ({imb['count']} examples)")

    # Coverage matrix table
    matrix = report["coverage_matrix"]
    if matrix:
        domains = sorted({d for counts in matrix.values() for d in counts})
        table = Table(title="Primitive x Domain Coverage")
        table.add_column("Primitive", style="cyan")
        for d in domains:
            table.add_column(d[:8], justify="center")
        table.add_column("Total", justify="right", style="bold")

        for primitive in sorted(matrix.keys()):
            row = [primitive]
            total = 0
            for d in domains:
                count = matrix[primitive].get(d, 0)
                total += count
                color = "green" if count > 0 else "red"
                row.append(f"[{color}]{count}[/]")
            row.append(str(total))
            table.add_row(*row)

        console.print(table)


def run_phase5(input_path: str | None = None) -> dict:
    """Run coverage validation and print report."""
    input_path = input_path or str(data_path("phase4", "filtered.jsonl"))
    pairs = read_jsonl_list(input_path, _FilteredPair)

    report = validate_coverage(pairs)
    print_coverage_report(report)

    # Save report
    import json

    report_path = data_path("phase5", "coverage_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Coverage report saved to {report_path}")
    return report
