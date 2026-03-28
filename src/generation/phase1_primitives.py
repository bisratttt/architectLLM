"""Phase 1: Extract primitives from blog posts using a teacher model."""

from __future__ import annotations

import asyncio
import logging

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.common.config import load_settings, load_prompt, data_path
from src.common.llm_client import LLMClient
from src.common.models import BlogPost, PrimitiveAnnotation
from src.common.storage import read_jsonl, append_jsonl, load_processed_ids

logger = logging.getLogger(__name__)


async def run_phase1(input_path: str | None = None, output_path: str | None = None) -> None:
    """Extract primitives from each blog post using GPT-4.1."""
    settings = load_settings()
    input_path = input_path or str(data_path("extracted", "posts.jsonl"))
    output_path = output_path or str(data_path("phase1", "annotations.jsonl"))

    prompt_template = load_prompt("extract_primitives")
    client = LLMClient(
        model=settings["teacher_model"],
        concurrency=settings["rate_limit"]["concurrency"],
    )

    # Load already-processed to enable resume
    processed_ids = load_processed_ids(output_path, id_field="post_id")
    posts = [p for p in read_jsonl(input_path, BlogPost) if p.id not in processed_ids]

    logger.info(f"Phase 1: Processing {len(posts)} posts ({len(processed_ids)} already done)")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Extracting primitives...", total=len(posts))

        async def _process(post: BlogPost):
            # Truncate markdown to ~4000 words to fit context
            content = " ".join(post.markdown.split()[:4000])
            user_prompt = prompt_template.replace("{content}", content)

            try:
                result = await client.complete_json(
                    system="You are a system design analysis expert. Respond only in valid JSON.",
                    user=user_prompt,
                    response_model=_Phase1Response,
                    temperature=settings["temperature"]["phase1_extraction"],
                )

                annotation = PrimitiveAnnotation(
                    post_id=post.id,
                    primary_primitive=result.primary_primitive,
                    secondary_primitives=result.secondary_primitives,
                    constraints=result.constraints,
                    tradeoffs=result.tradeoffs,
                    failure_modes=result.failure_modes,
                    domain_independent_lesson=result.domain_independent_lesson,
                    source_url=post.url,
                )
                append_jsonl(output_path, annotation)
            except Exception as e:
                logger.error(f"Failed to annotate post {post.id}: {e}")
            finally:
                progress.advance(task)

        await asyncio.gather(*[_process(p) for p in posts])

    logger.info(f"Phase 1 complete. {client.usage.summary()}")


# Internal response model for JSON parsing
from pydantic import BaseModel


class _Phase1Response(BaseModel):
    primary_primitive: str
    secondary_primitives: list[str] = []
    constraints: list[str] = []
    tradeoffs: list[str] = []
    failure_modes: list[str] = []
    domain_independent_lesson: str
