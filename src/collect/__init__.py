# Phase 1 headline collection -- see src/collect/run.py
from src.collect.base import BaseSource, HeadlineWriter, SourceConfig
from src.collect.rss import RSSSource
from src.collect.html import HTMLSource
from src.collect.sitemap import SitemapSource

__all__ = ["BaseSource", "HeadlineWriter", "SourceConfig", "RSSSource", "HTMLSource", "SitemapSource"]
