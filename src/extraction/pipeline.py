"""Orchestrator for the full content extraction pipeline."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import ssl

import aiohttp
import certifi
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.common.config import load_settings, data_path
from src.common.storage import append_jsonl, load_processed_ids
from src.extraction.discovery import discover_all_posts, fetch_opml
from src.extraction.extractor import fetch_and_extract

logger = logging.getLogger(__name__)


async def run_extraction(
    since: str | None = None,
    limit: int | None = None,
    max_concurrent: int = 10,
) -> None:
    """Run the full extraction pipeline: discover -> fetch -> extract -> store."""
    settings = load_settings()
    output_path = data_path("extracted", "posts.jsonl")

    # Load already-processed URLs to enable resume
    processed_ids = load_processed_ids(output_path)
    logger.info(f"Found {len(processed_ids)} already-processed posts")

    # Step 1: Discover posts from RSS feeds
    since_dt = datetime.fromisoformat(since) if since else None
    sources = await fetch_opml()
    posts = await discover_all_posts(sources, since=since_dt)

    # Filter out already-processed
    posts = [p for p in posts if p.id not in processed_ids]
    if limit:
        posts = posts[:limit]

    logger.info(f"Processing {len(posts)} new posts")

    # Step 2+3: Fetch and extract with bounded concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    delay = settings["extraction"]["request_delay_seconds"]
    extracted_count = 0
    skipped_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Extracting posts...", total=len(posts))

        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": "Mozilla/5.0 (compatible; architectLLM/0.1)"},
        ) as session:

            async def _process(post):
                nonlocal extracted_count, skipped_count
                async with semaphore:
                    result = await fetch_and_extract(
                        post,
                        session=session,
                        min_words=settings["extraction"]["min_word_count"],
                    )
                    if result:
                        append_jsonl(output_path, result)
                        extracted_count += 1
                    else:
                        skipped_count += 1
                    progress.advance(task)
                    await asyncio.sleep(delay)

            await asyncio.gather(*[_process(p) for p in posts])

    logger.info(
        f"Extraction complete: {extracted_count} extracted, {skipped_count} skipped"
    )
