"""Final export to JSONL and optional HuggingFace Dataset push."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.common.config import data_path
from src.common.models import TrainingExample, HarmonyExample
from src.common.storage import read_jsonl_list, write_jsonl
from src.formatting.harmony import single_turn_to_harmony, multi_turn_to_harmony
from src.formatting.composition import balance_dataset
from src.generation.phase4_filtering import _FilteredPair
from src.common.models import MultiTurnResponse

logger = logging.getLogger(__name__)


def export_dataset(
    filtered_path: str | None = None,
    multi_turn_path: str | None = None,
    output_path: str | None = None,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> Path:
    """Export the final balanced training dataset in Harmony JSONL format."""
    filtered_path = filtered_path or str(data_path("phase4", "filtered.jsonl"))
    multi_turn_path = multi_turn_path or str(data_path("phase3", "multi_turn_responses.jsonl"))
    output_path = output_path or str(data_path("final", "training_data.jsonl"))

    # Load filtered single-turn pairs
    pairs = read_jsonl_list(filtered_path, _FilteredPair)
    logger.info(f"Loaded {len(pairs)} filtered single-turn pairs")

    # Convert to TrainingExamples
    examples = []
    for pair in pairs:
        example = single_turn_to_harmony(
            question=pair.question,
            response=pair.response,
            quality_score=pair.quality_score,
        )
        examples.append(example)

    # Load and convert multi-turn responses
    mt_responses = read_jsonl_list(multi_turn_path, MultiTurnResponse)
    for mt in mt_responses:
        example = multi_turn_to_harmony(
            turns=mt.turns,
            primitive="",  # Would need to look up from question
            domain="",
            quality_score=0.0,
        )
        examples.append(example)

    logger.info(f"Total examples before balancing: {len(examples)}")

    # Balance to target composition
    balanced = balance_dataset(examples)

    # Write final JSONL - only the Harmony messages (what the model trains on)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for example in balanced:
            # Output format: {"messages": [{"role": ..., "content": ..., "channel": ...}, ...]}
            row = {
                "messages": [
                    {k: v for k, v in msg.model_dump().items() if v is not None}
                    for msg in example.harmony.messages
                ]
            }
            f.write(json.dumps(row) + "\n")

    logger.info(f"Exported {len(balanced)} training examples to {output}")

    # Also write a metadata sidecar
    meta_path = output.with_suffix(".meta.json")
    from collections import Counter

    complexity_dist = Counter(ex.complexity for ex in balanced)
    primitive_dist = Counter(ex.primitive for ex in balanced)
    domain_dist = Counter(ex.domain for ex in balanced)

    meta = {
        "total_examples": len(balanced),
        "complexity_distribution": dict(complexity_dist),
        "primitive_distribution": dict(primitive_dist),
        "domain_distribution": dict(domain_dist),
        "avg_quality_score": sum(ex.quality_score for ex in balanced) / max(len(balanced), 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: push to HuggingFace Hub
    if push_to_hub and hub_repo:
        try:
            from datasets import Dataset

            records = []
            with open(output) as f:
                for line in f:
                    records.append(json.loads(line))

            ds = Dataset.from_list(records)
            ds.push_to_hub(hub_repo)
            logger.info(f"Pushed dataset to HuggingFace Hub: {hub_repo}")
        except ImportError:
            logger.error("Install 'datasets' package to push to HuggingFace Hub")

    return output
