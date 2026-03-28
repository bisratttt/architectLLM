"""CLI entry point for the architectLLM training data pipeline."""

from __future__ import annotations

import asyncio
import logging

import click
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)


@click.group()
def main():
    """architectLLM - Training data pipeline for system design architect LLM."""
    pass


# === Part A: Extraction ===


@main.command()
@click.option("--since", default=None, help="Only fetch posts published after this date (ISO format)")
@click.option("--limit", default=None, type=int, help="Max number of posts to process")
@click.option("--concurrency", default=10, type=int, help="Max concurrent HTTP requests")
def extract(since, limit, concurrency):
    """Extract blog posts from engineering blog RSS feeds."""
    from src.extraction.pipeline import run_extraction

    asyncio.run(run_extraction(since=since, limit=limit, max_concurrent=concurrency))


# === Part B: Generation ===


@main.group()
def generate():
    """Run training data generation phases."""
    pass


@generate.command("phase1")
@click.option("--input", "input_path", default=None, help="Input JSONL path")
@click.option("--output", "output_path", default=None, help="Output JSONL path")
def gen_phase1(input_path, output_path):
    """Phase 1: Extract primitives from blog posts."""
    from src.generation.phase1_primitives import run_phase1

    asyncio.run(run_phase1(input_path=input_path, output_path=output_path))


@generate.command("phase2")
@click.option("--input", "input_path", default=None, help="Input JSONL path")
@click.option("--output", "output_path", default=None, help="Output JSONL path")
def gen_phase2(input_path, output_path):
    """Phase 2: Generate multi-domain questions via Evol-Instruct."""
    from src.generation.phase2_questions import run_phase2

    asyncio.run(run_phase2(input_path=input_path, output_path=output_path))


@generate.command("phase3")
@click.option("--input", "input_path", default=None, help="Input JSONL path")
@click.option("--output", "output_path", default=None, help="Output JSONL path")
def gen_phase3(input_path, output_path):
    """Phase 3: Generate expert responses with chain-of-thought."""
    from src.generation.phase3_responses import run_phase3

    asyncio.run(run_phase3(input_path=input_path, output_path=output_path))


@generate.command("phase4")
@click.option("--questions", "questions_path", default=None, help="Questions JSONL path")
@click.option("--responses", "responses_path", default=None, help="Responses JSONL path")
@click.option("--output", "output_path", default=None, help="Output JSONL path")
def gen_phase4(questions_path, responses_path, output_path):
    """Phase 4: Quality filtering (brand check, dedup, LLM judge)."""
    from src.generation.phase4_filtering import run_phase4

    asyncio.run(
        run_phase4(
            questions_path=questions_path,
            responses_path=responses_path,
            output_path=output_path,
        )
    )


@generate.command("phase5")
@click.option("--input", "input_path", default=None, help="Filtered JSONL path")
def gen_phase5(input_path):
    """Phase 5: Validate cross-domain coverage."""
    from src.generation.phase5_validation import run_phase5

    run_phase5(input_path=input_path)


# === Export ===


@main.command()
@click.option("--input", "filtered_path", default=None, help="Filtered JSONL path")
@click.option("--output", "output_path", default=None, help="Output JSONL path")
@click.option("--push-to-hub", is_flag=True, help="Push to HuggingFace Hub")
@click.option("--hub-repo", default=None, help="HuggingFace repo ID")
def export(filtered_path, output_path, push_to_hub, hub_repo):
    """Export balanced training dataset in Harmony format."""
    from src.formatting.export import export_dataset

    export_dataset(
        filtered_path=filtered_path,
        output_path=output_path,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
    )


# === Analysis ===


@main.command()
@click.option("--input", "dataset_path", default=None, help="Dataset JSONL path")
def stats(dataset_path):
    """Print dataset statistics and coverage analysis."""
    from src.analysis.coverage import print_dataset_stats

    print_dataset_stats(dataset_path=dataset_path)


if __name__ == "__main__":
    main()
