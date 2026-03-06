"""Web fetch tool for Axiom CLI.

Fetches a URL and returns its content as text. Optionally extracts
readable text from HTML using BeautifulSoup.
"""

from __future__ import annotations

from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

_MAX_CONTENT = 80_000  # max chars to return
_TIMEOUT = 30  # default request timeout in seconds
_MAX_REDIRECTS = 5

# User-Agent to avoid bot-blocking on common sites
_USER_AGENT = (
    "Mozilla/5.0 (compatible; AxiomCLI/1.0; +https://github.com/axiom-cli)"
)


class WebFetchTool(AxiomTool):
    """Fetch a URL and return its content as text/markdown."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return its content as text. "
        "Extracts readable text from HTML pages by default (strips nav, ads, scripts). "
        "Supports HTTP/HTTPS. Returns raw text for non-HTML content."
    )
    risk_level = "low"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch (must include http:// or https://)",
            },
            "extract_text": {
                "type": "boolean",
                "description": "Extract readable text from HTML (default: true)",
                "default": True,
            },
        },
        "required": ["url"],
    }

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]
        extract_text: bool = kwargs.get("extract_text", True)

        if not url.strip():
            raise ToolError("Empty URL", tool_name=self.name)

        # Auto-add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            import httpx
        except ImportError:
            raise ToolError(
                "httpx is not installed. Run: pip install httpx",
                tool_name=self.name,
            )

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=_MAX_REDIRECTS,
                timeout=httpx.Timeout(_TIMEOUT),
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                response = await client.get(url)
        except httpx.TimeoutException:
            raise ToolError(
                f"Request timed out after {_TIMEOUT}s: {url}",
                tool_name=self.name,
            )
        except httpx.TooManyRedirects:
            raise ToolError(
                f"Too many redirects (>{_MAX_REDIRECTS}): {url}",
                tool_name=self.name,
            )
        except httpx.ConnectError as exc:
            raise ToolError(
                f"Connection failed: {exc}",
                tool_name=self.name,
            )
        except httpx.HTTPError as exc:
            raise ToolError(
                f"HTTP error: {exc}",
                tool_name=self.name,
            )

        # Check status
        if response.status_code >= 400:
            raise ToolError(
                f"HTTP {response.status_code} for {url}: {response.reason_phrase}",
                tool_name=self.name,
            )

        content_type = response.headers.get("content-type", "")
        text = response.text

        # Extract readable text from HTML
        if extract_text and "text/html" in content_type:
            text = _extract_readable(text, url)

        # Truncate
        if len(text) > _MAX_CONTENT:
            text = text[:_MAX_CONTENT] + f"\n\n... [truncated at {_MAX_CONTENT:,} chars]"

        # Add metadata header
        header = f"[{response.status_code}] {url} ({len(text):,} chars)"
        if response.url != httpx.URL(url):
            header += f"\n[redirected to: {response.url}]"

        return f"{header}\n\n{text}"


def _extract_readable(html: str, url: str = "") -> str:
    """Extract readable text content from HTML, stripping boilerplate.

    Falls back to raw HTML if BeautifulSoup is not available.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: just strip tags with a simple regex
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                               "aside", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Try to find the main content area
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"class": "content"})
    )

    target = main if main else soup.body if soup.body else soup

    # Extract text with some structure
    lines: list[str] = []
    for element in target.stripped_strings:
        line = element.strip()
        if line:
            lines.append(line)

    # Also extract links
    links: list[str] = []
    for a in target.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if text and href and not href.startswith(("#", "javascript:")):
            links.append(f"  [{text}]({href})")

    result = "\n".join(lines)

    if links:
        # Append a small links section (max 20)
        link_section = "\n".join(links[:20])
        if len(links) > 20:
            link_section += f"\n  ... and {len(links) - 20} more links"
        result += f"\n\n--- Links ---\n{link_section}"

    return result
