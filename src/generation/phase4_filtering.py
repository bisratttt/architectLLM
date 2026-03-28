"""Phase 4: Quality filtering - brand check, dedup, length, LLM judge."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re

import tiktoken
from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.common.config import load_settings, load_brand_blocklist, load_prompt, data_path
from src.common.llm_client import LLMClient
from src.common.models import GeneratedQuestion, ExpertResponse, QualityScore
from src.common.storage import read_jsonl_list, write_jsonl, append_jsonl, load_processed_ids

logger = logging.getLogger(__name__)

# Patterns where brand mentions are acceptable (as examples, not recommendations)
ALLOWED_PATTERNS = [
    r"popular implementations include .+",
    r"options like .+",
    r"such as .+",
    r"examples include .+",
    r"tools like .+",
]


class _JudgeResponse(BaseModel):
    technical_accuracy: float
    completeness: float
    structure: float
    actionability: float
    primitive_coverage: float
    brand_leak: bool
    leads_with_pattern: bool
    overall_score: float
    reasoning: str


def _check_brand_leaks(text: str, blocklist: list[str]) -> tuple[bool, str]:
    """Check if text contains brand names outside of allowed patterns."""
    text_lower = text.lower()
    for brand in blocklist:
        # Use word boundaries to avoid false positives (e.g., "meta" in "metadata")
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, text_lower):
            # Check if it appears in an allowed context
            if any(re.search(p, text_lower) for p in ALLOWED_PATTERNS):
                continue
            return False, f"Brand leak: {brand}"

    return True, "OK"


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _find_near_duplicates(
    questions: list[GeneratedQuestion], threshold: float = 0.92
) -> dict[str, float]:
    """Find near-duplicate questions using Qwen3 embeddings + cosine similarity.
    Returns a dict mapping question_id -> max similarity score to any other question.
    """
    if len(questions) < 2:
        return {}

    import torch
    from pathlib import Path
    from src.common.config import data_path

    cache_dir = Path(data_path("phase4"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_cache = cache_dir / "embeddings.npy"
    ids_cache = cache_dir / "embedding_ids.json"

    question_ids = [q.id for q in questions]

    # Load cached embeddings if they match current questions
    if emb_cache.exists() and ids_cache.exists():
        import json as _json
        cached_ids = _json.loads(ids_cache.read_text())
        if cached_ids == question_ids:
            logger.info("Loading cached embeddings from disk...")
            all_embeddings = np.load(str(emb_cache))
            logger.info(f"Loaded {len(all_embeddings)} cached embeddings")
        else:
            logger.info("Cached embeddings stale (question list changed), re-computing...")
            all_embeddings = None
    else:
        all_embeddings = None

    if all_embeddings is None:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading Qwen3-Embedding-0.6B for semantic dedup...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        model.eval()

        texts = [q.question for q in questions]

        # Embed in batches on CPU
        emb_list = []
        batch_size = 32
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                emb_list.append(embeddings)
            if (batch_idx + 1) % 20 == 0:
                logger.info(f"Embedded {batch_idx + 1}/{total_batches} batches...")

        all_embeddings = torch.cat(emb_list, dim=0).float().numpy()

        # Save to disk
        np.save(str(emb_cache), all_embeddings)
        import json as _json
        ids_cache.write_text(_json.dumps(question_ids))
        logger.info(f"Saved {len(all_embeddings)} embeddings to {emb_cache}")

    logger.info(f"Embedded {len(all_embeddings)} questions, computing similarity...")

    # Compute cosine similarity matrix in chunks to save memory
    sim_max = np.zeros(len(questions))
    chunk_size = 500
    for i in range(0, len(all_embeddings), chunk_size):
        chunk = all_embeddings[i : i + chunk_size]
        sim_block = cosine_similarity(chunk, all_embeddings)
        # Zero out self-similarity
        for j in range(len(chunk)):
            sim_block[j, i + j] = 0
        sim_max[i : i + len(chunk)] = np.maximum(
            sim_max[i : i + len(chunk)], sim_block.max(axis=1)
        )

    max_sims = {}
    for idx, q in enumerate(questions):
        max_sims[q.id] = float(sim_max[idx])

    above = sum(1 for v in max_sims.values() if v >= threshold)
    logger.info(f"Semantic dedup: {above} questions above {threshold} threshold")

    return max_sims


async def run_phase4(
    questions_path: str | None = None,
    responses_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Run the full quality filtering pipeline."""
    settings = load_settings()
    filter_settings = settings["filtering"]

    questions_path = questions_path or str(data_path("phase2", "questions.jsonl"))
    responses_path = responses_path or str(data_path("phase3", "responses.jsonl"))
    output_path = output_path or str(data_path("phase4", "filtered.jsonl"))
    scores_path = str(data_path("phase4", "scores.jsonl"))

    blocklist = load_brand_blocklist()
    judge_prompt = load_prompt("llm_judge")

    # Load all data
    questions = read_jsonl_list(questions_path, GeneratedQuestion)
    responses = read_jsonl_list(responses_path, ExpertResponse)

    # Index responses by question_id
    response_map = {r.question_id: r for r in responses}

    # Only process questions that have responses
    questions = [q for q in questions if q.id in response_map]
    logger.info(f"Phase 4: Filtering {len(questions)} question-response pairs")

    # Step 1: Near-duplicate detection (batch, CPU-only)
    logger.info("Computing near-duplicates...")
    near_dup_scores = _find_near_duplicates(questions, filter_settings["near_duplicate_threshold"])

    # Step 2: Exact dedup
    seen_hashes = set()
    exact_dups = set()
    for q in questions:
        h = hashlib.sha256(q.question.strip().lower().encode()).hexdigest()
        if h in seen_hashes:
            exact_dups.add(q.id)
        seen_hashes.add(h)

    # Step 3: Brand check + length filter (CPU-only, fast)
    pre_filter_results = {}
    for q in questions:
        resp = response_map[q.id]
        passed_brand, brand_reason = _check_brand_leaks(resp.response, blocklist)
        token_count = _count_tokens(resp.response)
        is_dup = q.id in exact_dups
        near_dup_sim = near_dup_scores.get(q.id, 0.0)

        pre_filter_results[q.id] = {
            "passed_brand": passed_brand,
            "brand_reason": brand_reason,
            "token_count": token_count,
            "is_dup": is_dup,
            "near_dup_sim": near_dup_sim,
            "passed_length": token_count >= filter_settings["min_response_tokens"],
            "passed_near_dup": near_dup_sim < filter_settings["near_duplicate_threshold"],
        }

    # Pre-filter before expensive LLM judge
    candidates = [
        q
        for q in questions
        if pre_filter_results[q.id]["passed_brand"]
        and pre_filter_results[q.id]["passed_length"]
        and not pre_filter_results[q.id]["is_dup"]
        and pre_filter_results[q.id]["passed_near_dup"]
    ]

    logger.info(
        f"Pre-filter: {len(questions)} -> {len(candidates)} candidates "
        f"(brand: {sum(1 for q in questions if not pre_filter_results[q.id]['passed_brand'])}, "
        f"length: {sum(1 for q in questions if not pre_filter_results[q.id]['passed_length'])}, "
        f"dup: {len(exact_dups)}, "
        f"near-dup: {sum(1 for q in questions if not pre_filter_results[q.id]['passed_near_dup'])})"
    )

    # Step 4: LLM-as-judge scoring (optional — skip if no API credits)
    use_llm_judge = settings.get("use_llm_judge", True)

    if use_llm_judge:
        client = LLMClient(
            model=settings.get("judge_model", settings["teacher_model"]),
            concurrency=settings["rate_limit"]["concurrency"],
        )

        # Skip already-scored candidates (resume support)
        already_scored = load_processed_ids(scores_path, id_field="question_id")
        unscored = [q for q in candidates if q.id not in already_scored]
        logger.info(f"LLM judge: {len(unscored)} unscored candidates ({len(already_scored)} already scored)")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("LLM judge scoring...", total=len(unscored))

            async def _judge(q: GeneratedQuestion):
                resp = response_map[q.id]
                prompt = (
                    judge_prompt.replace("{question}", q.question)
                    .replace("{response}", resp.response)
                )

                try:
                    judge_result = await client.complete_json(
                        system="You are a system design training data quality evaluator. Respond only in valid JSON.",
                        user=prompt,
                        response_model=_JudgeResponse,
                        temperature=settings["temperature"]["phase4_judge"],
                    )

                    pre = pre_filter_results[q.id]
                    accepted = judge_result.overall_score >= filter_settings["judge_score_threshold"]

                    score = QualityScore(
                        question_id=q.id,
                        passed_brand_check=pre["passed_brand"],
                        token_count=pre["token_count"],
                        is_duplicate=pre["is_dup"],
                        nearest_neighbor_sim=pre["near_dup_sim"],
                        llm_judge_score=judge_result.overall_score,
                        accepted=accepted,
                    )
                    append_jsonl(scores_path, score)

                    if accepted:
                        append_jsonl(
                            output_path,
                            _FilteredPair(
                                question=q,
                                response=resp,
                                quality_score=judge_result.overall_score,
                            ),
                        )

                except Exception as e:
                    logger.error(f"Judge failed for {q.id}: {e}")
                finally:
                    progress.advance(task)

            await asyncio.gather(*[_judge(q) for q in unscored])

        logger.info(f"Phase 4 complete. {client.usage.summary()}")
    else:
        # Skip LLM judge — accept all candidates that passed local filters
        logger.info("Skipping LLM judge (use_llm_judge=false). Auto-accepting all candidates.")
        for q in candidates:
            resp = response_map[q.id]
            pre = pre_filter_results[q.id]

            score = QualityScore(
                question_id=q.id,
                passed_brand_check=pre["passed_brand"],
                token_count=pre["token_count"],
                is_duplicate=pre["is_dup"],
                nearest_neighbor_sim=pre["near_dup_sim"],
                llm_judge_score=0.0,
                accepted=True,
            )
            append_jsonl(scores_path, score)
            append_jsonl(
                output_path,
                _FilteredPair(
                    question=q,
                    response=resp,
                    quality_score=0.0,
                ),
            )

        logger.info(f"Phase 4 complete (local filters only). {len(candidates)} accepted.")


class _FilteredPair(BaseModel):
    question: GeneratedQuestion
    response: ExpertResponse
    quality_score: float
