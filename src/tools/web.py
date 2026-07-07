"""Web tools - fetch web pages and search the web."""

import re
from langchain_core.tools import tool


@tool
def web_fetch(url: str) -> str:
    """Fetch and extract text content from a web page URL.
    
    Use this to read documentation, API references, or any web content.
    
    Args:
        url: The full URL to fetch (must start with http:// or https://)
    
    Returns:
        Extracted text content from the page
    """
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    try:
        import httpx
        from bs4 import BeautifulSoup

        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; OpenCode-DeepAgents/1.0)",
                },
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse multiple blank lines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        # Truncate
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text) - max_chars} more chars)"
        return text if text.strip() else "(empty page)"
    except ImportError as e:
        return f"Missing dependency: {e}. Install with: pip install httpx beautifulsoup4"
    except Exception as e:
        return f"Error fetching {url}: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web and return relevant results.
    
    Use this to find current information, documentation, solutions to problems,
    or anything that requires up-to-date web knowledge.
    
    Args:
        query: The search query string
    
    Returns:
        Search results with titles, URLs, and snippets
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"No results found for: {query}"

        output = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            href = result.get("href", "No URL")
            body = result.get("body", "No description")
            output.append(f"{i}. {title}\n   URL: {href}\n   {body}")

        return "\n\n".join(output)
    except ImportError:
        return "Missing dependency: ddgs. Install with: pip install ddgs"
    except Exception as e:
        return f"Error searching: {e}"


def create_web_fetch_tool():
    return web_fetch


def create_web_search_tool():
    return web_search
