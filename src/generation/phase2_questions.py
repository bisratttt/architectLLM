"""Phase 2: Generate multi-domain questions using Evol-Instruct."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random

from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.common.config import (
    load_settings,
    load_prompt,
    load_domains,
    load_primitives,
    get_all_primitive_names,
    data_path,
)
from src.common.llm_client import LLMClient
from src.common.models import PrimitiveAnnotation, GeneratedQuestion, MultiTurnQuestion
from src.common.storage import read_jsonl, append_jsonl, load_processed_ids

logger = logging.getLogger(__name__)


class _EvolResponse(BaseModel):
    questions: list[_QuestionItem]


class _QuestionItem(BaseModel):
    domain: str
    question: str
    complexity: str


class _MultiTurnResponse(BaseModel):
    turns: list[str]


def _get_primitive_description(primitive_name: str) -> str:
    """Look up a primitive's description from the taxonomy."""
    primitives = load_primitives()
    for category in primitives.values():
        for p in category:
            if p["name"] == primitive_name:
                return p["description"]
    return primitive_name


async def run_phase2(input_path: str | None = None, output_path: str | None = None) -> None:
    """Generate multi-domain questions for each primitive annotation."""
    settings = load_settings()
    input_path = input_path or str(data_path("phase1", "annotations.jsonl"))
    output_path = output_path or str(data_path("phase2", "questions.jsonl"))
    multi_turn_output = str(data_path("phase2", "multi_turn_questions.jsonl"))

    evol_template = load_prompt("evol_instruct")
    multi_turn_template = load_prompt("multi_turn")
    all_domains = load_domains()
    gen_settings = settings["generation"]

    client = LLMClient(
        model=settings["teacher_model"],
        concurrency=settings["rate_limit"]["concurrency"],
    )

    valid_primitives = set(get_all_primitive_names())
    processed_ids = load_processed_ids(output_path, id_field="source_annotation_id")
    annotations = [
        a
        for a in read_jsonl(input_path, PrimitiveAnnotation)
        if a.post_id not in processed_ids
        and a.primary_primitive in valid_primitives
    ]

    logger.info(f"Phase 2: Generating questions for {len(annotations)} annotations")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Generating questions...", total=len(annotations))

        async def _process(annotation: PrimitiveAnnotation):
            # Pick random domains
            selected_domains = random.sample(
                all_domains, min(gen_settings["domains_per_primitive"], len(all_domains))
            )
            domain_names = [d["name"] for d in selected_domains]

            # Determine complexity distribution
            count = gen_settings["questions_per_annotation"]
            design_count = max(1, count - 4)  # Remaining after 1 each of other types

            prompt = (
                evol_template.replace("{primitive}", annotation.primary_primitive)
                .replace(
                    "{primitive_description}",
                    _get_primitive_description(annotation.primary_primitive),
                )
                .replace("{domain_independent_lesson}", annotation.domain_independent_lesson)
                .replace("{constraints}", ", ".join(annotation.constraints))
                .replace("{count}", str(count))
                .replace("{domains}", ", ".join(domain_names))
                .replace("{design_count}", str(design_count))
            )

            try:
                result = await client.complete_json(
                    system="You are an expert at generating diverse system design interview questions. Respond only in valid JSON.",
                    user=prompt,
                    response_model=_EvolResponse,
                    temperature=settings["temperature"]["phase2_questions"],
                )

                for i, q in enumerate(result.questions):
                    question_id = hashlib.sha256(
                        f"{annotation.post_id}:{i}:{q.question[:50]}".encode()
                    ).hexdigest()[:16]

                    generated = GeneratedQuestion(
                        id=question_id,
                        source_annotation_id=annotation.post_id,
                        primitive=annotation.primary_primitive,
                        domain=q.domain,
                        question=q.question,
                        complexity=q.complexity,
                        evol_generation=0,
                    )
                    append_jsonl(output_path, generated)

                # Generate one multi-turn conversation per annotation
                if result.questions:
                    initial_q = result.questions[0]
                    mt_prompt = (
                        multi_turn_template.replace("{primitive}", annotation.primary_primitive)
                        .replace("{domain}", initial_q.domain)
                        .replace("{initial_question}", initial_q.question)
                    )

                    mt_result = await client.complete_json(
                        system="You are a system design interviewer. Respond only in valid JSON.",
                        user=mt_prompt,
                        response_model=_MultiTurnResponse,
                        temperature=settings["temperature"]["phase2_questions"],
                    )

                    mt_id = hashlib.sha256(
                        f"mt:{annotation.post_id}".encode()
                    ).hexdigest()[:16]

                    multi_turn = MultiTurnQuestion(
                        id=mt_id,
                        source_annotation_id=annotation.post_id,
                        primitive=annotation.primary_primitive,
                        domain=initial_q.domain,
                        turns=[initial_q.question] + mt_result.turns,
                    )
                    append_jsonl(multi_turn_output, multi_turn)

            except Exception as e:
                logger.error(f"Failed to generate questions for {annotation.post_id}: {e}")
            finally:
                progress.advance(task)

        await asyncio.gather(*[_process(a) for a in annotations])

    logger.info(f"Phase 2 complete. {client.usage.summary()}")
