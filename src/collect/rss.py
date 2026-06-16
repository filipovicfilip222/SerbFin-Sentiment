"""
RSS/Atom feed collector (feedparser).

Primary collection method -- use for any source that exposes a feed.
Stores only: title -> text, link -> url, published -> date, source name.
No article bodies are fetched or stored (legal constraint, SPEC.md §4.1).

Verified working feeds (as of 2026-05-31):
  - https://biznis.rs/feed/
  - https://www.b92.net/rss/biz
  - https://novaekonomija.rs/feed/
"""

from __future__ import annotations

import sys
from datetime import date

import feedparser
import requests

from src.collect.base import BaseSource, USER_AGENT


def _entry_date(entry) -> str:
    """Extract ISO date from a feedparser entry; fall back to today."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return date(t.tm_year, t.tm_mon, t.tm_mday).isoformat()
            except Exception:
                pass
    return date.today().isoformat()


class RSSSource(BaseSource):
    """Collect headlines from a single RSS/Atom feed."""

    def collect(self) -> list[dict]:
        url = self.config.feed_url
        print(f"  [RSS] {self.config.name} -> {url}")
        self._throttle(url)

        # Fetch via requests so we control encoding; pass bytes to feedparser
        # so it can detect the charset itself (avoids feedparser's latin-1 fallback
        # choking on Serbian characters like em-dashes in titles).
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=20,
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except requests.HTTPError as exc:
            print(f"  [RSS] HTTP error for {self.config.name}: {exc}", file=sys.stderr)
            return []
        except Exception as exc:
            print(f"  [RSS] Error fetching {self.config.name}: {exc}", file=sys.stderr)
            return []

        # bozo=True means feedparser detected a malformed feed, but may still have entries
        if feed.get("bozo") and not feed.entries:
            print(
                f"  [RSS] Feed parse error for {self.config.name}: "
                f"{feed.get('bozo_exception', 'unknown')}",
                file=sys.stderr,
            )
            return []

        records = []
        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue
            records.append(
                {
                    "text": title,
                    "source": self.config.name,
                    "url": link,
                    "date": _entry_date(entry),
                    "lang": "sr",
                }
            )

        print(f"  [RSS] {self.config.name}: {len(records)} items")
        return records
