"""Content extraction using Trafilatura."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import trafilatura

from src.common.models import BlogPost
from src.extraction.discovery import DiscoveredPost

logger = logging.getLogger(__name__)


def extract_blog_post(html: str, post: DiscoveredPost, min_words: int = 300) -> BlogPost | None:
    """Extract and clean blog post content from HTML.

    Returns None if the content is too short or extraction fails.
    """
    # Trafilatura can output Markdown directly
    markdown = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        include_links=True,
        output_format="markdown",
        favor_recall=True,
    )

    if not markdown:
        logger.debug(f"Trafilatura extraction failed for {post.url}")
        return None

    # Word count filter
    word_count = len(markdown.split())
    if word_count < min_words:
        logger.debug(f"Post too short ({word_count} words): {post.url}")
        return None

    # Extract metadata via bare_extraction (returns a Document object)
    meta = trafilatura.bare_extraction(html)
    title = post.title
    author = None
    date = post.date

    if meta:
        title = getattr(meta, "title", None) or post.title
        author = getattr(meta, "author", None)
        date = getattr(meta, "date", None) or post.date

    return BlogPost(
        id=hashlib.sha256(post.url.encode()).hexdigest(),
        url=post.url,
        title=title,
        author=author,
        date=date,
        source_blog=post.source_blog,
        categories=post.categories or [],
        markdown=markdown,
        word_count=word_count,
        crawled_at=datetime.now(timezone.utc).isoformat(),
    )


async def fetch_and_extract(
    post: DiscoveredPost,
    session=None,
    min_words: int = 300,
) -> BlogPost | None:
    """Fetch HTML from a URL and extract the blog post content."""
    import aiohttp

    try:
        if session:
            async with session.get(post.url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    logger.debug(f"HTTP {resp.status} for {post.url}")
                    return None
                html = await resp.text()
        else:
            async with aiohttp.ClientSession() as s:
                async with s.get(post.url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()

        return extract_blog_post(html, post, min_words=min_words)

    except Exception as e:
        logger.warning(f"Failed to fetch/extract {post.url}: {e}")
        return None
