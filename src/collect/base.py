"""
Core abstractions for headline collection.

  RawRecord      — imported from src.data.schema (unlabeled record)
  SourceConfig   — parsed from configs/sources.yaml
  HeadlineWriter — dedup-aware JSONL appender, validates via RawRecord
  BaseSource     — rate-limited abstract base for all collectors
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.data.schema import RawRecord

USER_AGENT = (
    "SerbFinSentimentCollector/1.0 "
    "(academic NLP research -- serbfin-sentiment project; "
    "headline text + URL only, no full articles)"
)
DEFAULT_DELAY = 2.0  # seconds between requests to the same domain


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


@dataclass
class SourceConfig:
    """One entry from configs/sources.yaml."""

    name: str
    type: str           # "rss" | "html" | "sitemap"
    enabled: bool = True
    note: str = ""
    # RSS
    feed_url: str = ""
    # HTML
    list_url: str = ""
    headline_selector: str = ""
    href_pattern: str = ""   # optional regex; keeps only matching hrefs
    base_url: str = ""
    pages: int = 1
    pagination: str = ""     # template with {page}, appended after list_url
    # Sitemap
    sitemap_url: str = ""
    sitemap_child_pattern: str = "post-sitemap"  # regex to select child sitemaps
    max_child_sitemaps: int = 5    # how many child sitemaps to download (newest first)
    max_fetch: int = 200           # cap on page fetches for plain (non-news) sitemaps

    @classmethod
    def from_dict(cls, d: dict) -> "SourceConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class HeadlineWriter:
    """
    Append-only, dedup-aware writer for data/raw/headlines.jsonl.

    Dedup keys: normalized text AND url — either match ⟹ skip.
    Sequential IDs (sfs-NNNNNN) continue from the highest existing ID.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._seen_urls: set[str] = set()
        self._seen_texts: set[str] = set()
        self._counter: int = 0
        self._written: int = 0
        self._skipped: int = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url = rec.get("url", "").strip()
                text = rec.get("text", "").strip()
                if url:
                    self._seen_urls.add(url)
                if text:
                    self._seen_texts.add(_normalize(text))
                rec_id = rec.get("id", "")
                if rec_id.startswith("sfs-"):
                    try:
                        self._counter = max(self._counter, int(rec_id[4:]))
                    except ValueError:
                        pass

    def try_write(self, raw: dict) -> bool:
        """
        Validate, dedup, and append. Returns True if written.

        raw must contain: text, source, url, date.
        label / label_source must be absent (this is the raw phase).
        """
        text = raw.get("text", "").strip()
        url = raw.get("url", "").strip()

        if not text:
            self._skipped += 1
            return False

        norm = _normalize(text)
        if (url and url in self._seen_urls) or norm in self._seen_texts:
            self._skipped += 1
            return False

        self._counter += 1
        payload = dict(raw)
        payload["id"] = f"sfs-{self._counter:06d}"
        payload.setdefault("lang", "sr")

        # Remove any label fields that shouldn't be in raw records
        payload.pop("label", None)
        payload.pop("label_source", None)

        try:
            RawRecord.model_validate(payload)
        except Exception:
            self._skipped += 1
            self._counter -= 1
            return False

        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

        self._seen_urls.add(url)
        self._seen_texts.add(norm)
        self._written += 1
        return True

    @property
    def written(self) -> int:
        return self._written

    @property
    def skipped(self) -> int:
        return self._skipped

    @property
    def total_in_file(self) -> int:
        return self._counter

    @property
    def seen_urls(self) -> frozenset[str]:
        """Read-only snapshot of all URLs already present in the output file."""
        return frozenset(self._seen_urls)


class BaseSource(ABC):
    """Rate-limited, polite base class for all source collectors."""

    def __init__(self, config: SourceConfig, delay: float = DEFAULT_DELAY) -> None:
        self.config = config
        self._delay = delay
        self._last_by_domain: dict[str, float] = {}

    def _throttle(self, url: str) -> None:
        """Sleep if needed to respect per-domain rate limit."""
        domain = urlparse(url).netloc
        elapsed = time.monotonic() - self._last_by_domain.get(domain, 0.0)
        wait = self._delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_by_domain[domain] = time.monotonic()

    @abstractmethod
    def collect(self) -> list[dict]:
        """Return list of raw record dicts (text, source, url, date, lang)."""
        ...
