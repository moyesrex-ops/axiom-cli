"""Multi-source research tool for deep web research.

Aggregates results from Tavily, DuckDuckGo, GitHub API, and raw web fetch
to produce comprehensive research reports with citations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional

import httpx

from axiom.core.tools.base import AxiomTool, ToolError

logger = logging.getLogger(__name__)


class ResearchTool(AxiomTool):
    """Deep multi-source web research tool.

    Searches across multiple sources (Tavily, DuckDuckGo, GitHub, Exa)
    and synthesizes results into a structured report with citations.
    """

    name = "research"
    description = (
        "Perform deep web research across multiple sources. "
        "Returns structured results with sources and citations. "
        "Modes: 'quick' (single-source fast), 'deep' (multi-source cross-referenced), "
        "'github' (code/repo search), 'academic' (papers/research)."
    )
    risk_level = "low"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The research query or question",
            },
            "mode": {
                "type": "string",
                "description": "Research mode: quick, deep, github, academic",
                "enum": ["quick", "deep", "github", "academic"],
                "default": "deep",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results per source (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        mode = kwargs.get("mode", "deep")
        max_results = kwargs.get("max_results", 5)

        if not query.strip():
            raise ToolError("Empty research query", tool_name=self.name)

        all_results: list[dict[str, Any]] = []
        errors: list[str] = []

        if mode == "github":
            results = await _search_github(query, max_results)
            all_results.extend(results)
        elif mode == "quick":
            # Single source: Tavily first, then DuckDuckGo fallback
            results = await _search_tavily(query, max_results)
            if results:
                all_results.extend(results)
            else:
                results = await _search_duckduckgo(query, max_results)
                all_results.extend(results)
        else:
            # Deep mode: fan out to all sources in parallel
            tasks = [
                _search_tavily(query, max_results),
                _search_duckduckgo(query, max_results),
            ]

            # Add GitHub if it seems code-related
            code_keywords = {"code", "library", "framework", "api", "sdk", "package", "repo"}
            if any(kw in query.lower() for kw in code_keywords):
                tasks.append(_search_github(query, max_results))

            results_groups = await asyncio.gather(*tasks, return_exceptions=True)
            for group in results_groups:
                if isinstance(group, Exception):
                    errors.append(str(group))
                elif isinstance(group, list):
                    all_results.extend(group)

        if not all_results:
            if errors:
                return f"Research failed. Errors:\n" + "\n".join(errors)
            return f"No results found for: {query}"

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_results: list[dict[str, Any]] = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)
            elif not url:
                unique_results.append(r)

        # Format output
        output = [f"## Research Results: {query}\n"]
        output.append(f"Found **{len(unique_results)}** results across {_count_sources(unique_results)} sources.\n")

        for i, r in enumerate(unique_results[:max_results * 2], 1):
            source = r.get("source", "web")
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = r.get("snippet", "")[:500]

            output.append(f"### {i}. {title}")
            if url:
                output.append(f"**Source**: [{source}] {url}")
            if snippet:
                output.append(f"{snippet}")
            output.append("")

        if errors:
            output.append(f"\n*Some sources returned errors: {'; '.join(errors[:3])}*")

        return "\n".join(output)


def _count_sources(results: list[dict[str, Any]]) -> int:
    """Count unique source types."""
    return len(set(r.get("source", "web") for r in results))


# ── Source: Tavily ──────────────────────────────────────────────────


async def _search_tavily(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search via Tavily API (structured search with extraction)."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        logger.debug("Tavily API key not set, skipping")
        return []

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "advanced",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[dict[str, Any]] = []

        # Include the direct answer if available
        answer = data.get("answer")
        if answer:
            results.append({
                "title": "Tavily AI Answer",
                "url": "",
                "snippet": answer,
                "source": "tavily",
            })

        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": "tavily",
            })

        return results[:max_results]

    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return []


# ── Source: DuckDuckGo ──────────────────────────────────────────────


async def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search via DuckDuckGo Instant Answer API (free, no key needed)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_redirect": "1",
                    "no_html": "1",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[dict[str, Any]] = []

        # Abstract (main answer)
        abstract = data.get("AbstractText", "")
        if abstract:
            results.append({
                "title": data.get("Heading", "DuckDuckGo Answer"),
                "url": data.get("AbstractURL", ""),
                "snippet": abstract,
                "source": "duckduckgo",
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                    "source": "duckduckgo",
                })

        # If no results from instant answer, try the html search
        if not results:
            results = await _search_ddg_html(query, max_results)

        return results[:max_results]

    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []


async def _search_ddg_html(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Fallback DuckDuckGo search via HTML scraping."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            resp.raise_for_status()
            html = resp.text

        # Simple extraction of result links and snippets
        results: list[dict[str, Any]] = []
        # Look for result blocks
        import re
        blocks = re.findall(
            r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        for url, title, snippet in blocks[:max_results]:
            # Clean HTML tags
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "duckduckgo",
                })

        return results
    except Exception:
        return []


# ── Source: GitHub ──────────────────────────────────────────────────


async def _search_github(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search GitHub repositories and code."""
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    results: list[dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Search repositories
            resp = await client.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": max_results,
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            for repo in data.get("items", []):
                results.append({
                    "title": f"{repo['full_name']} ⭐{repo.get('stargazers_count', 0)}",
                    "url": repo.get("html_url", ""),
                    "snippet": (
                        f"{repo.get('description', 'No description')}\n"
                        f"Language: {repo.get('language', 'N/A')} | "
                        f"Stars: {repo.get('stargazers_count', 0)} | "
                        f"Forks: {repo.get('forks_count', 0)}"
                    ),
                    "source": "github",
                })

    except Exception as e:
        logger.warning("GitHub search failed: %s", e)

    return results[:max_results]
