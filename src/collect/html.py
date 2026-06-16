"""
Polite fallback HTML scraper for sources that do not expose RSS feeds.

Rules (SPEC.md §4.1):
  - Check robots.txt before fetching any URL; skip disallowed paths.
  - Fetch listing pages only -- never article bodies.
  - Store: headline text, source URL, collection date.
  - Polite User-Agent identifies the project.
  - Per-domain rate limiting (>= 2 s between requests).

Verified sources and selectors (as of 2026-05-31):
  eKapija   -- https://www.ekapija.com/news/finansije
               CSS: a[href*="/news/"], href_pattern: /news/\\d+
               Pagination: ?page={page}   (verified working)
               Date: visible as DD.MM.YYYY. plain text near each article

  Kamatica  -- https://www.kamatica.com/vesti/
               CSS: h4 a, href_pattern: /vest/
               Pagination: ?page={page}  (returns same page -- dedup handles it)

  BGDX      -- https://bgdx.rs/trzista_i_hartije/vesti/izdavaoci
               CSS: a[href*="/vesti/"], href_pattern: /vesti/\\d+
               Pagination: ?page={page}  (verified different items)
"""

from __future__ import annotations

import re
import sys
from datetime import date
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from selectolax.parser import HTMLParser

from src.collect.base import BaseSource, SourceConfig, USER_AGENT

# Matches Serbian DD.MM.YYYY date strings (with or without trailing dot)
_DATE_RE = re.compile(r"(\d{1,2})\.(\d{2})\.(\d{4})")


def _extract_date(text: str) -> str | None:
    """Convert DD.MM.YYYY from surrounding text to ISO YYYY-MM-DD."""
    m = _DATE_RE.search(text)
    if m:
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(y, mo, d).isoformat()
        except ValueError:
            pass
    return None


class _RobotsCache:
    """Per-origin robots.txt cache (one RobotFileParser per scheme+netloc).

    Fetches robots.txt using the project User-Agent (via requests) so sites
    that return 403 to bare Python-urllib still get the real file parsed.
    A 403 from the project UA is treated as disallow-all; any other fetch
    error (timeout, DNS) is treated as allow (be polite, not paranoid).
    """

    def __init__(self) -> None:
        self._parsers: dict[str, RobotFileParser] = {}

    def _fetch_robots(self, robots_url: str) -> RobotFileParser:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            resp = requests.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=10,
                allow_redirects=True,
            )
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in (401, 403):
                rp.disallow_all = True
            # 404 / other 4xx → allow_all (standard convention)
        except Exception:
            pass  # unreachable → assume allowed
        return rp

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._parsers:
            self._parsers[origin] = self._fetch_robots(f"{origin}/robots.txt")
        return self._parsers[origin].can_fetch(USER_AGENT, url)


_robots = _RobotsCache()


class HTMLSource(BaseSource):
    """Scrape headline links from a paginated listing page."""

    def __init__(self, config: SourceConfig, delay: float = 2.0) -> None:
        super().__init__(config, delay)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "sr,sr-Latn;q=0.9,en;q=0.5",
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        self._href_re = re.compile(config.href_pattern) if config.href_pattern else None

    def _page_url(self, page: int) -> str:
        if page == 1:
            return self.config.list_url
        pat = self.config.pagination
        if not pat:
            return self.config.list_url
        if "{page}" in pat:
            sep = "&" if "?" in self.config.list_url else "?"
            return self.config.list_url + sep + pat.format(page=page)
        return self.config.list_url

    def _fetch(self, url: str) -> str | None:
        self._throttle(url)
        try:
            resp = self._session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.text
            print(f"  [HTML] HTTP {resp.status_code} for {url}", file=sys.stderr)
        except Exception as exc:
            print(f"  [HTML] Request error for {url}: {exc}", file=sys.stderr)
        return None

    def _parse_listing(self, html: str, page_url: str) -> list[dict]:
        """Extract (text, url, date) tuples from one listing page."""
        tree = HTMLParser(html)
        results: list[dict] = []
        seen_hrefs: set[str] = set()

        for node in tree.css(self.config.headline_selector):
            # The matched node may already be an <a>, or we look for a child <a>
            anchor = node if node.tag == "a" else node.css_first("a")
            if anchor is None:
                continue

            href = (anchor.attributes.get("href") or "").strip()
            title = anchor.text(strip=True)

            if not href or not title or len(title) < 8:
                continue

            # Apply per-source href filter
            if self._href_re and not self._href_re.search(href):
                continue

            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)

            full_url = href if href.startswith("http") else urljoin(page_url, href)
            if not full_url.startswith("http"):
                continue

            # Try to read a date from the surrounding card text
            article_date = None
            parent = anchor.parent
            if parent:
                card_text = parent.text(strip=True)
                article_date = _extract_date(card_text)
                # Walk up one more level if the date wasn't in the immediate parent
                if article_date is None and parent.parent:
                    article_date = _extract_date(parent.parent.text(strip=True))
            if article_date is None:
                article_date = date.today().isoformat()

            results.append(
                {
                    "text": title,
                    "source": self.config.name,
                    "url": full_url,
                    "date": article_date,
                    "lang": "sr",
                }
            )

        return results

    def collect(self) -> list[dict]:
        all_records: list[dict] = []

        for page in range(1, self.config.pages + 1):
            url = self._page_url(page)

            if not _robots.allowed(url):
                print(f"  [HTML] robots.txt disallows {url} -- stopping", file=sys.stderr)
                break

            print(f"  [HTML] {self.config.name} page {page} -> {url}")
            html = self._fetch(url)
            if not html:
                break

            records = self._parse_listing(html, url)
            if not records:
                # No headlines found -- either last page or selector mismatch
                print(
                    f"  [HTML] No headlines on page {page} -- stopping pagination",
                    file=sys.stderr,
                )
                break

            all_records.extend(records)

        print(f"  [HTML] {self.config.name}: {len(all_records)} items fetched")
        return all_records
