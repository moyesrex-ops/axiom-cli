"""HTTP request tool for Axiom CLI.

Performs arbitrary HTTP requests (GET, POST, PUT, PATCH, DELETE)
with JSON/form payloads, custom headers, and configurable timeouts.

Unlike ``web_fetch`` (which is read-only and extracts text from HTML),
this tool is designed for API interaction — sending payloads, reading
JSON responses, and working with REST/GraphQL endpoints.
"""

from __future__ import annotations

import json
from typing import Any

from axiom.core.tools.base import AxiomTool, ToolError

_DEFAULT_TIMEOUT = 30
_MAX_RESPONSE = 50_000  # max chars in response body to return
_USER_AGENT = "AxiomCLI/1.0"


class HTTPTool(AxiomTool):
    """Perform HTTP requests to APIs and web endpoints."""

    name = "http_request"
    description = (
        "Send an HTTP request (GET/POST/PUT/PATCH/DELETE) to a URL. "
        "Supports JSON and form-encoded payloads, custom headers, and "
        "returns the response status, headers, and body. "
        "Use for API calls, webhooks, and programmatic HTTP interactions."
    )
    risk_level = "medium"
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
                "description": "HTTP method",
                "default": "GET",
            },
            "url": {
                "type": "string",
                "description": "Target URL (must include http:// or https://)",
            },
            "headers": {
                "type": "object",
                "description": "Custom request headers (key-value pairs)",
                "default": {},
            },
            "json_body": {
                "type": "object",
                "description": "JSON payload (sets Content-Type to application/json)",
            },
            "form_body": {
                "type": "object",
                "description": "Form-encoded payload (sets Content-Type to application/x-www-form-urlencoded)",
            },
            "raw_body": {
                "type": "string",
                "description": "Raw text/binary body (use when json_body and form_body don't apply)",
            },
            "timeout": {
                "type": "integer",
                "description": f"Request timeout in seconds (default: {_DEFAULT_TIMEOUT})",
                "default": _DEFAULT_TIMEOUT,
            },
            "follow_redirects": {
                "type": "boolean",
                "description": "Whether to follow HTTP redirects (default: true)",
                "default": True,
            },
        },
        "required": ["url"],
    }

    async def execute(self, **kwargs: Any) -> str:
        method = kwargs.get("method", "GET").upper()
        url: str = kwargs["url"]
        headers: dict = kwargs.get("headers", {})
        json_body = kwargs.get("json_body")
        form_body = kwargs.get("form_body")
        raw_body = kwargs.get("raw_body")
        timeout = kwargs.get("timeout", _DEFAULT_TIMEOUT)
        follow_redirects = kwargs.get("follow_redirects", True)

        if not url.strip():
            raise ToolError("Empty URL", tool_name=self.name)

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            import httpx
        except ImportError:
            raise ToolError(
                "httpx is not installed. Run: pip install httpx",
                tool_name=self.name,
            )

        # Build request kwargs
        req_kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": {"User-Agent": _USER_AGENT, **headers},
            "timeout": httpx.Timeout(timeout),
            "follow_redirects": follow_redirects,
        }

        # Set body — only one type allowed
        body_types = [b for b in (json_body, form_body, raw_body) if b is not None]
        if len(body_types) > 1:
            raise ToolError(
                "Provide at most one of: json_body, form_body, raw_body",
                tool_name=self.name,
            )

        if json_body is not None:
            req_kwargs["json"] = json_body
        elif form_body is not None:
            req_kwargs["data"] = form_body
        elif raw_body is not None:
            req_kwargs["content"] = raw_body

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(**req_kwargs)
        except httpx.TimeoutException:
            raise ToolError(
                f"Request timed out after {timeout}s: {method} {url}",
                tool_name=self.name,
            )
        except httpx.ConnectError as exc:
            raise ToolError(f"Connection failed: {exc}", tool_name=self.name)
        except httpx.HTTPError as exc:
            raise ToolError(f"HTTP error: {exc}", tool_name=self.name)

        # Build response summary
        parts: list[str] = []

        # Status line
        parts.append(f"HTTP {response.status_code} {response.reason_phrase}")
        parts.append(f"URL: {response.url}")

        # Response headers (abbreviated)
        interesting_headers = [
            "content-type", "content-length", "location",
            "x-ratelimit-remaining", "retry-after", "set-cookie",
        ]
        header_lines = []
        for h in interesting_headers:
            if h in response.headers:
                header_lines.append(f"  {h}: {response.headers[h]}")
        if header_lines:
            parts.append("Headers:\n" + "\n".join(header_lines))

        # Response body
        body_text = response.text
        content_type = response.headers.get("content-type", "")

        # Try to pretty-print JSON responses
        if "application/json" in content_type:
            try:
                parsed = response.json()
                body_text = json.dumps(parsed, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, ValueError):
                pass

        if len(body_text) > _MAX_RESPONSE:
            body_text = body_text[:_MAX_RESPONSE] + f"\n\n... [truncated at {_MAX_RESPONSE:,} chars]"

        parts.append(f"Body ({len(body_text):,} chars):\n{body_text}")

        return "\n\n".join(parts)
