"""
Sitemap-based historical bulk collector (SPEC.md Phase 1 volume unlock).

Two operating modes, chosen automatically per-sitemap:

  CHEAP  -- Google News sitemap (biznis.rs/news-sitemap.xml)
            news:title + news:publication_date present in the XML.
            Zero extra HTTP fetches needed for headline text.

  FETCH  -- Plain sitemap (post-sitemap*.xml on biznis.rs, novaekonomija.rs)
            Only <loc> + <lastmod> present.
            Fetches each article page to read og:title / <title>.
            Rate-limited (default 1.5 s/request), capped at max_fetch per run.
            Most recent URLs (by lastmod desc) fetched first.

Legal constraints respected (SPEC.md ss4.1):
  - Only headline text + url + date stored; NO article body.
  - robots.txt checked before every page fetch.
  - Polite User-Agent identifies the project.

Config fields used (in configs/sources.yaml, type: sitemap):
  sitemap_url:            root sitemap or sitemap index URL
  sitemap_child_pattern:  regex to select child sitemaps from an index
                          (e.g. "post-sitemap" matches post-sitemap*.xml)
  max_child_sitemaps:     how many child sitemaps to download (newest first
                          by numeric suffix; default 5)
  max_fetch:              cap on page-fetch operations per run (default 200;
                          override with --max-fetch N on the CLI)
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from datetime import date
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from selectolax.parser import HTMLParser

from src.collect.base import BaseSource, SourceConfig, USER_AGENT

# --------------------------------------------------------------------------- #
# XML namespace constants
# --------------------------------------------------------------------------- #
_NS_SM = "http://www.sitemaps.org/schemas/sitemap/0.9"
_NS_NEWS = "http://www.google.com/schemas/sitemap-news/0.9"


def _t(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _child_text(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _iso_date(s: str) -> str:
    """Extract YYYY-MM-DD from an ISO 8601 or ISO-date string; fall back to today."""
    if s and len(s) >= 10:
        candidate = s[:10]
        try:
            date.fromisoformat(candidate)
            return candidate
        except ValueError:
            pass
    return date.today().isoformat()


def _numeric_key(url: str) -> int:
    """Return the trailing integer from a URL (for sitemap sorting). 0 if absent."""
    m = re.search(r"(\d+)(?:\.\w+)?$", url)
    return int(m.group(1)) if m else 0


# --------------------------------------------------------------------------- #
# robots.txt cache (module-level, one parser per origin)
# --------------------------------------------------------------------------- #
class _RobotsCache:
    def __init__(self) -> None:
        self._parsers: dict[str, RobotFileParser] = {}

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._parsers:
            rp = RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                pass
            self._parsers[origin] = rp
        return self._parsers[origin].can_fetch(USER_AGENT, url)


_robots = _RobotsCache()


# --------------------------------------------------------------------------- #
# SitemapSource
# --------------------------------------------------------------------------- #
class SitemapSource(BaseSource):
    """
    Collects historical headlines from XML sitemaps.

    Google News sitemaps supply titles for free.
    Plain sitemaps require a per-article og:title fetch (capped at max_fetch).
    """

    def __init__(
        self,
        config: SourceConfig,
        delay: float = 1.5,
        max_fetch: Optional[int] = None,
        max_child_sitemaps: Optional[int] = None,
    ) -> None:
        super().__init__(config, delay)
        # CLI flags override config values when explicitly provided
        self._max_fetch = config.max_fetch if max_fetch is None else max_fetch
        self._max_child = config.max_child_sitemaps if max_child_sitemaps is None else max_child_sitemaps
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "sr,sr-Latn;q=0.9,en;q=0.5",
        })

    # ── HTTP helpers ────────────────────────────────────────────────────────────

    def _get(self, url: str, timeout: int = 20) -> Optional[requests.Response]:
        self._throttle(url)
        try:
            resp = self._session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            print(f"  [SM] HTTP {resp.status_code}: {url}", file=sys.stderr)
        except Exception as exc:
            print(f"  [SM] Fetch error {url}: {exc}", file=sys.stderr)
        return None

    def _fetch_xml(self, url: str) -> Optional[ET.Element]:
        resp = self._get(url)
        if resp is None:
            return None
        try:
            return ET.fromstring(resp.content)
        except ET.ParseError as exc:
            print(f"  [SM] XML parse error {url}: {exc}", file=sys.stderr)
            return None

    # ── Sitemap discovery ────────────────────────────────────────────────────────

    def _child_sitemaps(self, root: ET.Element) -> list[str]:
        """
        Extract child sitemap URLs from a <sitemapindex>, filtered by
        sitemap_child_pattern and limited to max_child_sitemaps.

        Sort priority (highest = newest content first):
          1. <lastmod> from the index entry (ISO date desc) -- most reliable.
             Sites that declare lastmod per child sitemap update it when new
             articles are added, so the most recently modified sitemap has
             the freshest content.
          2. Numeric suffix desc as a tie-breaker (standard Yoast ascending
             numbering: higher number = newer batch of posts).

        This two-key sort works correctly even when sites have inverted or
        mixed numbering schemes (e.g. Biznis.rs lowest-numbered = newest).
        """
        pat_str = self.config.sitemap_child_pattern
        pat = re.compile(pat_str) if pat_str else None
        entries: list[tuple[str, str]] = []  # (loc, lastmod)
        for sm in root.findall(_t(_NS_SM, "sitemap")):
            loc = _child_text(sm, _t(_NS_SM, "loc"))
            lastmod = _child_text(sm, _t(_NS_SM, "lastmod"))
            if loc and (pat is None or pat.search(loc)):
                entries.append((loc, lastmod))

        # Sort: lastmod desc primary, numeric key desc secondary
        entries.sort(key=lambda e: (e[1], _numeric_key(e[0])), reverse=True)
        return [loc for loc, _ in entries[: self._max_child]]

    def _is_index(self, root: ET.Element) -> bool:
        tag = root.tag
        return tag == _t(_NS_SM, "sitemapindex") or root.find(_t(_NS_SM, "sitemap")) is not None

    # ── Entry parsing ────────────────────────────────────────────────────────────

    def _parse_entries(self, root: ET.Element) -> list[dict]:
        """
        Parse <url> elements from a single sitemap.

        Returns dicts with keys: url, date, and either
          title + free=True  (Google News sitemap)
          free=False         (plain sitemap, title TBD via page fetch)
        """
        entries: list[dict] = []
        for url_elem in root.findall(_t(_NS_SM, "url")):
            loc = _child_text(url_elem, _t(_NS_SM, "loc"))
            if not loc:
                continue

            lastmod = _child_text(url_elem, _t(_NS_SM, "lastmod"))
            entry_date = _iso_date(lastmod)

            # Google News block -- title and date present in XML
            news = url_elem.find(_t(_NS_NEWS, "news"))
            if news is not None:
                title = _child_text(news, _t(_NS_NEWS, "title"))
                pub_date = _child_text(news, _t(_NS_NEWS, "publication_date"))
                if pub_date:
                    entry_date = _iso_date(pub_date)
                if title:
                    entries.append({"url": loc, "date": entry_date, "title": title, "free": True})
                    continue

            entries.append({"url": loc, "date": entry_date, "free": False})

        return entries

    # ── Page fetch for plain sitemaps ────────────────────────────────────────────

    def _page_title(self, url: str) -> Optional[str]:
        """Fetch an article page; extract og:title or <title>. Returns None on failure."""
        if not _robots.allowed(url):
            return None
        resp = self._get(url, timeout=15)
        if resp is None:
            return None
        try:
            tree = HTMLParser(resp.content)
            # og:title is authoritative and clean
            og = tree.css_first('meta[property="og:title"]')
            if og:
                t = (og.attributes.get("content") or "").strip()
                if len(t) >= 8:
                    return t
            # Fallback: <title>, strip " - SiteName" suffixes
            node = tree.css_first("title")
            if node:
                t = node.text(strip=True)
                for sep in (" - ", " | ", " -- ", " : "):
                    if sep in t:
                        t = t.split(sep)[0].strip()
                        break
                if len(t) >= 8:
                    return t
        except Exception:
            pass
        return None

    # ── Main collect ──────────────────────────────────────────────────────────────

    def collect(self, skip_urls: frozenset[str] | None = None) -> list[dict]:
        """
        Collect headlines from this sitemap source.

        skip_urls: URLs already present in the output file.  Pass this to skip
                   re-fetching articles we already have, avoiding wasted HTTP
                   requests on repeat runs.  HeadlineWriter._seen_urls is the
                   right value to pass.
        """
        sitemap_url = self.config.sitemap_url
        print(f"  [SM] {self.config.name} -> {sitemap_url}")

        root = self._fetch_xml(sitemap_url)
        if root is None:
            return []

        if self._is_index(root):
            child_urls = self._child_sitemaps(root)
            print(
                f"  [SM] {self.config.name}: index -> {len(child_urls)} child sitemaps "
                f"(pattern={self.config.sitemap_child_pattern!r}, max={self._max_child})"
            )
            child_roots: list[ET.Element] = []
            for curl in child_urls:
                cr = self._fetch_xml(curl)
                if cr is not None:
                    child_roots.append(cr)
        else:
            child_roots = [root]

        # Collect all candidate entries
        all_entries: list[dict] = []
        for cr in child_roots:
            all_entries.extend(self._parse_entries(cr))

        free = [e for e in all_entries if e.get("free")]
        all_to_fetch = [e for e in all_entries if not e.get("free")]

        # Drop URLs we already have -- avoids wasting page-fetches on repeat runs
        if skip_urls:
            skipped_early = sum(1 for e in all_to_fetch if e["url"] in skip_urls)
            all_to_fetch = [e for e in all_to_fetch if e["url"] not in skip_urls]
        else:
            skipped_early = 0

        to_fetch = sorted(all_to_fetch, key=lambda e: e["date"], reverse=True)

        print(
            f"  [SM] {self.config.name}: {len(free)} free (news:title), "
            f"{len(to_fetch)} queued for page-fetch "
            f"({skipped_early} already-known skipped, cap={self._max_fetch})"
        )

        records: list[dict] = []

        # Free records -- no HTTP fetch
        for e in free:
            records.append({
                "text": e["title"],
                "source": self.config.name,
                "url": e["url"],
                "date": e["date"],
                "lang": "sr",
            })

        # Page-fetch records (capped)
        fetched_ok = 0
        fetch_errors = 0
        for e in to_fetch:
            if fetched_ok >= self._max_fetch:
                break
            title = self._page_title(e["url"])
            if title:
                records.append({
                    "text": title,
                    "source": self.config.name,
                    "url": e["url"],
                    "date": e["date"],
                    "lang": "sr",
                })
                fetched_ok += 1
            else:
                fetch_errors += 1
                # Don't abort on errors; keep going

        print(
            f"  [SM] {self.config.name}: {len(records)} records collected "
            f"({len(free)} free + {fetched_ok} page-fetched, {fetch_errors} errors)"
        )
        return records
