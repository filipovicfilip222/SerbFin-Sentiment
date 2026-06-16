"""
Phase 1 collection runner.

Reads configs/sources.yaml, runs each enabled source, appends to
data/raw/headlines.jsonl, and prints per-source + total stats.

Usage:
    # Daily RSS/HTML top-ups (default):
    python -m src.collect.run
    python -m src.collect.run --source "Biznis.rs"   # single source

    # Bulk historical harvest (run once):
    python -m src.collect.run --sitemap
    python -m src.collect.run --sitemap --max-fetch 200   # faster, fewer fetches

    # Inspection:
    python -m src.collect.run --dry-run
    python -m src.collect.run --sitemap --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from src.collect.base import HeadlineWriter, SourceConfig
from src.collect.html import HTMLSource
from src.collect.rss import RSSSource
from src.collect.sitemap import SitemapSource

TARGET = 2_000


def _load_config(path: Path) -> tuple[dict, list[SourceConfig]]:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    global_cfg = raw.get("global", {})
    sources = [SourceConfig.from_dict(s) for s in raw.get("sources", [])]
    return global_cfg, sources


def _make_source(
    cfg: SourceConfig,
    delay: float,
    max_fetch: Optional[int] = None,
    max_child_sitemaps: Optional[int] = None,
):
    if cfg.type == "rss":
        return RSSSource(cfg, delay=delay)
    if cfg.type == "html":
        return HTMLSource(cfg, delay=delay)
    if cfg.type == "sitemap":
        return SitemapSource(
            cfg,
            delay=delay,
            max_fetch=max_fetch,
            max_child_sitemaps=max_child_sitemaps,
        )
    raise ValueError(f"Unknown source type {cfg.type!r} for {cfg.name!r}")


def _type_filter(cfg: SourceConfig, sitemap_mode: bool) -> bool:
    if sitemap_mode:
        return cfg.type == "sitemap"
    return cfg.type in ("rss", "html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Serbian financial headlines")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sources.yaml"),
        help="Path to sources YAML (default: configs/sources.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output JSONL path from config",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Run only this source by name (within the active mode)",
    )
    parser.add_argument(
        "--sitemap",
        action="store_true",
        help="Run sitemap sources (bulk historical). Default: run RSS/HTML sources.",
    )
    parser.add_argument(
        "--max-fetch",
        type=int,
        default=None,
        metavar="N",
        help="Override max_fetch for all sitemap sources in this run (default: per-source config)",
    )
    parser.add_argument(
        "--max-child-sitemaps",
        type=int,
        default=None,
        metavar="N",
        help="Override max_child_sitemaps for all sitemap sources in this run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run and exit without fetching",
    )
    args = parser.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")

    global_cfg, sources = _load_config(args.config)
    delay = float(global_cfg.get("request_delay", 2.0))
    output_path = args.output or Path(global_cfg.get("output", "data/raw/headlines.jsonl"))

    mode = "sitemap" if args.sitemap else "rss/html"
    enabled = [s for s in sources if s.enabled and _type_filter(s, args.sitemap)]
    if args.source:
        enabled = [s for s in enabled if s.name == args.source]
        if not enabled:
            avail = [s.name for s in sources if s.enabled and _type_filter(s, args.sitemap)]
            sys.exit(
                f"Source {args.source!r} not found in {mode} mode. Available: {avail}"
            )

    print(f"Mode         : {mode}")
    print(f"Output file  : {output_path}")
    print(f"Request delay: {delay}s per domain")
    if args.sitemap and args.max_fetch is not None:
        print(f"max-fetch    : {args.max_fetch} (CLI override)")
    print(f"Active sources: {len(enabled)}")
    for s in sources:
        in_mode = _type_filter(s, args.sitemap)
        if not in_mode:
            continue
        status = "[on] " if s.enabled else "[off]"
        note = f"  [{s.note}]" if s.note and not s.enabled else ""
        extra = ""
        if s.type == "sitemap" and s.enabled:
            mf = args.max_fetch if args.max_fetch is not None else s.max_fetch
            mc = args.max_child_sitemaps if args.max_child_sitemaps is not None else s.max_child_sitemaps
            extra = f"  [max_fetch={mf}, max_child={mc}]"
        print(f"  {status} {s.name} ({s.type}){extra}{note}")
    print()

    if args.dry_run:
        print("Dry run -- exiting without fetching.")
        return

    writer = HeadlineWriter(output_path)
    existing_before = writer.total_in_file

    per_source: list[tuple[str, dict]] = []  # list to preserve order with duplicate names

    for cfg in enabled:
        src = _make_source(
            cfg,
            delay,
            max_fetch=args.max_fetch,
            max_child_sitemaps=args.max_child_sitemaps,
        )
        # Pass already-known URLs so sitemap sources skip re-fetching them
        if isinstance(src, SitemapSource):
            records = src.collect(skip_urls=writer.seen_urls)
        else:
            records = src.collect()

        written = 0
        for rec in records:
            if writer.try_write(rec):
                written += 1

        per_source.append((cfg.name, {
            "fetched": len(records),
            "written": written,
            "deduped": len(records) - written,
        }))

    # Summary table
    print()
    print("-" * 60)
    print(f"{'Source':<28}  {'Fetched':>7}  {'Written':>7}  {'Dedup':>5}")
    print("-" * 60)
    for name, stats in per_source:
        print(
            f"{name:<28}  {stats['fetched']:>7}  {stats['written']:>7}  {stats['deduped']:>5}"
        )
    print("-" * 60)

    total_in_file = writer.total_in_file
    newly_written = total_in_file - existing_before

    print(f"{'New records this run':<28}  {newly_written:>7}")
    print(f"{'Total in file':<28}  {total_in_file:>7}")
    print()

    if total_in_file < TARGET:
        pct = total_in_file / TARGET * 100
        print(
            f"Progress toward {TARGET:,} target: {total_in_file:,} / {TARGET:,} "
            f"({pct:.0f}%). Run again to accumulate more."
        )
    else:
        print(
            f"Target reached: {total_in_file:,} records >= {TARGET:,}. "
            "Ready for Phase 2 -- silver labeling."
        )

    if not args.sitemap:
        print(
            f"\nTip: run with --sitemap for a bulk historical harvest "
            f"(~1,000+ new records in one pass)."
        )
    print(
        f"\nNext step: python -m src.label.silver "
        f"--input {output_path} --output data/interim/silver.jsonl"
    )


if __name__ == "__main__":
    main()
