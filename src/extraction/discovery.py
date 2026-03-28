"""RSS feed discovery via OPML parsing and sitemap fallback."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

import ssl

import aiohttp
import certifi
import feedparser


def _make_ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())

logger = logging.getLogger(__name__)

OPML_URL = (
    "https://raw.githubusercontent.com/kilimchoi/engineering-blogs/master/engineering_blogs.opml"
)


@dataclass
class FeedSource:
    name: str
    feed_url: str
    site_url: str | None = None


@dataclass
class DiscoveredPost:
    url: str
    title: str
    source_blog: str
    date: str | None = None
    categories: list[str] | None = None

    @property
    def id(self) -> str:
        return hashlib.sha256(self.url.encode()).hexdigest()


async def fetch_opml(url: str = OPML_URL) -> list[FeedSource]:
    """Parse the kilimchoi/engineering-blogs OPML file for RSS feed URLs."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            text = await resp.text()

    root = ET.fromstring(text)
    sources = []
    for outline in root.iter("outline"):
        xml_url = outline.get("xmlUrl")
        if xml_url:
            sources.append(
                FeedSource(
                    name=outline.get("text", outline.get("title", "Unknown")),
                    feed_url=xml_url,
                    site_url=outline.get("htmlUrl"),
                )
            )
    logger.info(f"Discovered {len(sources)} feed sources from OPML")
    return sources


def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    parsed = urlparse(url)
    # Strip tracking params, trailing slashes
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


async def fetch_feed_posts(
    source: FeedSource,
    since: datetime | None = None,
    session: aiohttp.ClientSession | None = None,
) -> list[DiscoveredPost]:
    """Fetch article URLs from a single RSS feed."""
    posts = []
    try:
        if session:
            async with session.get(
                source.feed_url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                content = await resp.text()
        else:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    source.feed_url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    content = await resp.text()

        feed = feedparser.parse(content)
        for entry in feed.entries:
            # Filter by date if specified
            if since and hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
                if pub_date < since:
                    continue

            url = entry.get("link", "")
            if not url:
                continue

            date_str = None
            if hasattr(entry, "published"):
                date_str = entry.published

            categories = []
            if hasattr(entry, "tags"):
                categories = [t.term for t in entry.tags if hasattr(t, "term")]

            posts.append(
                DiscoveredPost(
                    url=_normalize_url(url),
                    title=entry.get("title", "Untitled"),
                    source_blog=source.name,
                    date=date_str,
                    categories=categories,
                )
            )
    except Exception as e:
        logger.warning(f"Failed to fetch feed {source.name} ({source.feed_url}): {e}")

    return posts


async def discover_all_posts(
    sources: list[FeedSource] | None = None,
    since: datetime | None = None,
    max_concurrent: int = 20,
) -> list[DiscoveredPost]:
    """Fetch posts from all sources with bounded concurrency."""
    if sources is None:
        sources = await fetch_opml()

    semaphore = asyncio.Semaphore(max_concurrent)
    all_posts = []

    connector = aiohttp.TCPConnector(ssl=_make_ssl_context())
    async with aiohttp.ClientSession(connector=connector) as session:

        async def _fetch(source: FeedSource):
            async with semaphore:
                return await fetch_feed_posts(source, since=since, session=session)

        results = await asyncio.gather(*[_fetch(s) for s in sources], return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            all_posts.extend(result)
        elif isinstance(result, Exception):
            logger.warning(f"Feed fetch error: {result}")

    # Deduplicate by normalized URL
    seen = set()
    unique = []
    for post in all_posts:
        normalized = _normalize_url(post.url)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(post)

    logger.info(f"Discovered {len(unique)} unique posts from {len(sources)} sources")
    return unique
