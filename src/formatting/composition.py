"""Dataset composition balancing to hit target distribution."""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from src.common.config import load_settings
from src.common.models import TrainingExample

logger = logging.getLogger(__name__)


def balance_dataset(examples: list[TrainingExample], seed: int = 42) -> list[TrainingExample]:
    """Sample from the full pool to hit composition targets.

    Targets (from settings.yaml):
    - 30% single_primitive (1,500)
    - 25% full_system (1,250)
    - 20% cross_domain (1,000)
    - 10% tradeoff (500)
    - 10% failure (500)
    - 5% multi_turn (250)
    """
    settings = load_settings()
    comp = settings["composition"]
    total = comp["total"]

    targets = {
        "single_primitive": int(total * comp["single_primitive_pct"]),
        "full_system": int(total * comp["full_system_pct"]),
        "cross_domain": int(total * comp["cross_domain_pct"]),
        "tradeoff": int(total * comp["tradeoff_pct"]),
        "failure": int(total * comp["failure_pct"]),
        "multi_turn": int(total * comp["multi_turn_pct"]),
    }

    # Group by complexity
    by_complexity = defaultdict(list)
    for ex in examples:
        by_complexity[ex.complexity].append(ex)

    rng = random.Random(seed)
    selected = []

    for complexity, target_count in targets.items():
        pool = by_complexity.get(complexity, [])
        # Sort by quality score descending, then sample for diversity
        pool.sort(key=lambda x: x.quality_score, reverse=True)

        if len(pool) >= target_count:
            # Take top quality, but ensure primitive/domain diversity
            selected.extend(_diverse_sample(pool, target_count, rng))
        else:
            logger.warning(
                f"Insufficient {complexity} examples: {len(pool)}/{target_count}. "
                f"Using all {len(pool)}."
            )
            selected.extend(pool)

    logger.info(
        f"Balanced dataset: {len(selected)} examples "
        f"(target: {total}, from pool of {len(examples)})"
    )

    # Log distribution
    dist = defaultdict(int)
    for ex in selected:
        dist[ex.complexity] += 1
    for complexity, count in sorted(dist.items()):
        logger.info(f"  {complexity}: {count} ({count/max(len(selected),1)*100:.1f}%)")

    return selected


def _diverse_sample(
    pool: list[TrainingExample], n: int, rng: random.Random
) -> list[TrainingExample]:
    """Sample n items ensuring primitive and domain diversity.

    Strategy: round-robin across primitives, within each primitive round-robin domains.
    Fall back to quality-sorted top-n if diversity constraints can't be met.
    """
    # Group by primitive
    by_primitive = defaultdict(list)
    for ex in pool:
        by_primitive[ex.primitive].append(ex)

    # Round-robin across primitives
    selected = []
    primitive_keys = list(by_primitive.keys())
    rng.shuffle(primitive_keys)

    idx = 0
    while len(selected) < n:
        key = primitive_keys[idx % len(primitive_keys)]
        candidates = by_primitive[key]
        if candidates:
            # Pick from least-represented domain first
            domain_counts = defaultdict(int)
            for s in selected:
                if s.primitive == key:
                    domain_counts[s.domain] += 1

            # Sort candidates by domain count (ascending) then quality (descending)
            candidates.sort(
                key=lambda x: (domain_counts.get(x.domain, 0), -x.quality_score)
            )
            selected.append(candidates.pop(0))

        idx += 1
        # Safety: if we've cycled through all primitives and they're all empty, break
        if all(len(by_primitive[k]) == 0 for k in primitive_keys):
            break

    return selected[:n]
