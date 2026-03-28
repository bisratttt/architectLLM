"""Coverage analysis and dataset statistics."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.common.config import data_path
from src.common.models import TrainingExample

logger = logging.getLogger(__name__)
console = Console()


def print_dataset_stats(dataset_path: str | None = None) -> None:
    """Print comprehensive stats about the final training dataset."""
    dataset_path = dataset_path or str(data_path("final", "training_data.jsonl"))
    meta_path = Path(dataset_path).with_suffix(".meta.json")

    if not meta_path.exists():
        console.print("[red]No metadata file found. Run export first.[/]")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    console.print(f"\n[bold]Dataset Statistics[/bold]")
    console.print(f"Total examples: {meta['total_examples']}")
    console.print(f"Avg quality score: {meta['avg_quality_score']:.2f}")

    # Complexity distribution
    table = Table(title="Complexity Distribution")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")

    total = meta["total_examples"]
    for comp, count in sorted(meta["complexity_distribution"].items()):
        table.add_row(comp, str(count), f"{count/max(total,1)*100:.1f}%")
    console.print(table)

    # Top primitives
    table = Table(title="Top 15 Primitives")
    table.add_column("Primitive", style="cyan")
    table.add_column("Count", justify="right")

    sorted_prims = sorted(meta["primitive_distribution"].items(), key=lambda x: -x[1])
    for prim, count in sorted_prims[:15]:
        table.add_row(prim, str(count))
    console.print(table)

    # Domain distribution
    table = Table(title="Domain Distribution")
    table.add_column("Domain", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")

    for domain, count in sorted(meta["domain_distribution"].items(), key=lambda x: -x[1]):
        table.add_row(domain, str(count), f"{count/max(total,1)*100:.1f}%")
    console.print(table)
