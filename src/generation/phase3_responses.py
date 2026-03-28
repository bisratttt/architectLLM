"""Phase 3: Generate expert responses with chain-of-thought (analysis + final)."""

from __future__ import annotations

import asyncio
import logging

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.common.config import load_settings, load_prompt, data_path
from src.common.llm_client import LLMClient
from src.common.models import (
    PrimitiveAnnotation,
    GeneratedQuestion,
    MultiTurnQuestion,
    ExpertResponse,
    MultiTurnResponse,
    MultiTurnTurn,
)
from src.common.storage import read_jsonl, append_jsonl, load_processed_ids

logger = logging.getLogger(__name__)


def _build_source_context(
    annotation: PrimitiveAnnotation | None,
    blog_post: BlogPost | None,
) -> dict[str, str]:
    """Build source context dict from annotation and blog post."""
    if not annotation:
        return {
            "primitive": "unknown",
            "constraints": "none",
            "tradeoffs": "none",
            "failure_modes": "none",
            "lesson": "none",
            "blog_excerpt": "No source blog available.",
        }

    # Truncate blog to ~2000 words to fit in context
    excerpt = ""
    if blog_post and blog_post.markdown:
        words = blog_post.markdown.split()
        excerpt = " ".join(words[:2000])

    return {
        "primitive": annotation.primary_primitive,
        "constraints": ", ".join(annotation.constraints) if annotation.constraints else "none specified",
        "tradeoffs": ", ".join(annotation.tradeoffs) if annotation.tradeoffs else "none specified",
        "failure_modes": ", ".join(annotation.failure_modes) if annotation.failure_modes else "none specified",
        "lesson": annotation.domain_independent_lesson,
        "blog_excerpt": excerpt or "No blog content available.",
    }


def _fill_prompt(template: str, context: dict[str, str], **extra: str) -> str:
    """Fill a prompt template with context and extra fields."""
    result = template
    for key, value in {**context, **extra}.items():
        result = result.replace(f"{{{key}}}", value)
    return result


async def run_phase3(input_path: str | None = None, output_path: str | None = None) -> None:
    """Generate expert responses: analysis (CoT) + final answer for each question."""
    settings = load_settings()
    input_path = input_path or str(data_path("phase2", "questions.jsonl"))
    multi_turn_input = str(data_path("phase2", "multi_turn_questions.jsonl"))
    output_path = output_path or str(data_path("phase3", "responses.jsonl"))
    multi_turn_output = str(data_path("phase3", "multi_turn_responses.jsonl"))

    analysis_prompt = load_prompt("generate_analysis")
    response_prompt = load_prompt("generate_response")
    persona = load_prompt("system_architect_persona")

    client = LLMClient(
        model=settings["teacher_model"],
        concurrency=settings["rate_limit"]["concurrency"],
    )

    # Load annotations for grounding (constraints, tradeoffs, failure modes, lesson)
    annotations_path = str(data_path("phase1", "annotations.jsonl"))
    annotations = {a.post_id: a for a in read_jsonl(annotations_path, PrimitiveAnnotation)}

    logger.info(f"Loaded {len(annotations)} annotations for grounding")

    # Process single-turn questions
    processed_ids = load_processed_ids(output_path, id_field="question_id")
    questions = [
        q for q in read_jsonl(input_path, GeneratedQuestion) if q.id not in processed_ids
    ]

    logger.info(f"Phase 3: Generating responses for {len(questions)} questions")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Generating responses...", total=len(questions))

        async def _process_single(question: GeneratedQuestion):
            try:
                # Look up source annotation
                annotation = annotations.get(question.source_annotation_id)
                context = _build_source_context(annotation, None)

                # Pass 1: Generate analysis grounded in source material
                analysis_user = _fill_prompt(
                    analysis_prompt, context, question=question.question
                )
                chain_of_thought = await client.complete(
                    system="You are writing expert architectural analysis for a system design training dataset. Ground your analysis in the source material provided.",
                    user=analysis_user,
                    temperature=settings["temperature"]["phase3_analysis"],
                    max_tokens=2048,
                )

                # Pass 2: Generate structured response grounded in source + analysis
                response_user = _fill_prompt(
                    response_prompt, context,
                    question=question.question,
                    analysis=chain_of_thought or "No analysis available.",
                )
                final_response = await client.complete(
                    system=persona,
                    user=response_user,
                    temperature=settings["temperature"]["phase3_response"],
                    max_tokens=4096,
                )

                # Skip empty responses
                if not (chain_of_thought or "").strip() and not (final_response or "").strip():
                    logger.warning(f"Empty response for {question.id}, skipping")
                    return

                expert = ExpertResponse(
                    question_id=question.id,
                    chain_of_thought=chain_of_thought or "",
                    response=final_response or "",
                )
                append_jsonl(output_path, expert)

            except Exception as e:
                logger.error(f"Failed to generate response for {question.id}: {e}")
            finally:
                progress.advance(task)

        # Process in small batches — each response writes immediately within the batch
        batch_size = 10
        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            await asyncio.gather(*[_process_single(q) for q in batch])

    # Process multi-turn questions
    mt_processed = load_processed_ids(multi_turn_output, id_field="question_id")
    mt_questions = [
        q for q in read_jsonl(multi_turn_input, MultiTurnQuestion) if q.id not in mt_processed
    ]

    if mt_questions:
        logger.info(f"Phase 3: Generating multi-turn responses for {len(mt_questions)} conversations")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Multi-turn responses...", total=len(mt_questions))

            async def _process_multi(mt: MultiTurnQuestion):
                try:
                    # Look up source annotation for grounding
                    annotation = annotations.get(mt.source_annotation_id)
                    context = _build_source_context(annotation, None)
                    context_summary = (
                        f"Source primitive: {context['primitive']}\n"
                        f"Constraints: {context['constraints']}\n"
                        f"Key lesson: {context['lesson']}"
                    )

                    turns = []
                    messages = [{"role": "system", "content": persona + f"\n\nGROUNDING CONTEXT:\n{context_summary}"}]

                    for turn_question in mt.turns:
                        messages.append({"role": "user", "content": turn_question})

                        # Generate analysis grounded in source
                        prev_context = messages[-3]['content'] if len(messages) > 2 else 'none'
                        analysis = await client.complete(
                            system="You are writing architectural analysis for a training dataset. Ground your analysis in the source material.",
                            user=f"Source context: {context_summary}\n\nPrevious discussion: {prev_context}\n\nFollow-up question: {turn_question}",
                            temperature=settings["temperature"]["phase3_analysis"],
                            max_tokens=4096,
                        )

                        # Generate response
                        response = await client.complete_multi_turn(
                            messages=messages,
                            temperature=settings["temperature"]["phase3_response"],
                            max_tokens=8192,
                        )

                        turns.append(
                            MultiTurnTurn(
                                user=turn_question,
                                chain_of_thought=analysis or "",
                                response=response or "",
                            )
                        )
                        messages.append({"role": "assistant", "content": response or ""})

                    mt_response = MultiTurnResponse(question_id=mt.id, turns=turns)
                    append_jsonl(multi_turn_output, mt_response)

                except Exception as e:
                    logger.error(f"Failed multi-turn for {mt.id}: {e}")
                finally:
                    progress.advance(task)

            batch_size = 10
            for i in range(0, len(mt_questions), batch_size):
                batch = mt_questions[i : i + batch_size]
                await asyncio.gather(*[_process_multi(q) for q in batch])

    logger.info(f"Phase 3 complete. {client.usage.summary()}")
