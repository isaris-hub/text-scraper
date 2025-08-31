# scraper.py
#!/usr/bin/env python3
"""
Async website scraper & crawler.

Fix in this version
-------------------
Some sites publish a very permissive robots.txt like:

    User-agent: *
    Allow: /
    Sitemap: https://www.example.com/sitemap.xml

Python's built-in `urllib.robotparser` can still return False in certain cases
(e.g., odd encodings, UA matching quirks, redirects to a different host, or
vendor-specific directives). This version adds a **robust robots layer**:

- Robots are cached **per host** (netloc).
- If the robots text contains an **Allow: /** for either your configured
  user-agent token or `*`, we treat it as **allow-all** for that host.
- If `can_fetch(user_agent, url)` returns False, we fall back to `can_fetch("*", url)`.
- If parsing fails or the server returns a non-text robots file, we default to
  **polite allow** (do not block the crawl).
- After redirects, we re-check robots against the **final host**.

The rest of the scraper implements:
- BFS crawling within scope (registrable domain, optional subdomains)
- Sitemap seeding
- HTML text extraction to per-page .txt files
- PDF/Word download (in-scope only)
- External URL collection
- Strong logging & retry/backoff
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import gzip
import hashlib
import logging
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib import robotparser
from urllib.parse import urljoin, urlsplit, urlunsplit, urlencode, parse_qsl

import httpx
import tldextract
import yaml
from bs4 import BeautifulSoup
from slugify import slugify


# --------------------------- Configuration --------------------------------- #


@dataclasses.dataclass(frozen=True)
class Config:
    results_dir: str = "results"
    user_agent: str = "AsyncScraperBot/1.0 (+https://example.com/bot)"
    concurrency: int = 5
    delay: float = 1.0  # seconds between requests (base)
    timeout: int = 20  # seconds per request
    max_pages: int = 300
    max_depth: int = 5
    respect_robots: bool = True
    include_subdomains: bool = False
    same_registrable_domain: bool = True
    sites: tuple[str, ...] = ()

    @staticmethod
    def from_yaml(path: Path) -> "Config":
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Config(
            results_dir=data.get("results_dir", "results"),
            user_agent=data.get("user_agent", "AsyncScraperBot/1.0 (+https://example.com/bot)"),
            concurrency=int(data.get("concurrency", 5)),
            delay=float(data.get("delay", 1.0)),
            timeout=int(data.get("timeout", 20)),
            max_pages=int(data.get("max_pages", 300)),
            max_depth=int(data.get("max_depth", 5)),
            respect_robots=bool(data.get("respect_robots", True)),
            include_subdomains=bool(data.get("include_subdomains", False)),
            same_registrable_domain=bool(data.get("same_registrable_domain", True)),
            sites=tuple(data.get("sites", []) or ()),
        )


# ----------------------------- Utilities ----------------------------------- #


def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]


def file_safe_slug(text: str, maxlen: int = 80) -> str:
    s = slugify(text, max_length=maxlen, allow_unicode=False).strip("-_.")
    return s or sha1_short(text)


def normalize_url(url: str, base: Optional[str] = None, sort_query: bool = True) -> str:
    if base:
        url = urljoin(base, url)
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme not in ("http", "https"):
        return url
    if "@" in netloc:
        netloc = netloc.split("@", 1)[-1]
    if (scheme == "http" and netloc.endswith(":80")) or (scheme == "https" and netloc.endswith(":443")):
        netloc = netloc.rsplit(":", 1)[0]
    fragment = ""
    path = parts.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = parts.query
    if sort_query and query:
        q = parse_qsl(query, keep_blank_values=True)
        q.sort()
        query = urlencode(q, doseq=True)
    return urlunsplit((scheme, netloc, path, query, fragment))


def is_probably_html(url: str, content_type: Optional[str]) -> bool:
    if content_type and "text/html" in content_type.lower():
        return True
    return bool(re.search(r"\.(?:x?html?)$", url.split("?")[0], flags=re.I))


def detect_doc_kind(url: str, content_type: Optional[str]) -> Optional[str]:
    if content_type:
        ct = content_type.lower().split(";", 1)[0].strip()
        if ct == "application/pdf":
            return "pdf"
        if ct == "application/msword":
            return "doc"
        if ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return "docx"
    path = url.split("?", 1)[0].lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith(".doc"):
        return "doc"
    if path.endswith(".docx"):
        return "docx"
    return None


def get_registrable_domain(url: str) -> Tuple[str, str, str]:
    ext = tldextract.extract(url)
    return ext.subdomain, ext.domain, ext.suffix


def in_same_scope(url: str, site_root: str, include_subdomains: bool, same_registrable_domain: bool) -> bool:
    try:
        if urlsplit(url).scheme not in ("http", "https"):
            return False
        s_sub, s_dom, s_suf = get_registrable_domain(site_root)
        u_sub, u_dom, u_suf = get_registrable_domain(url)
        if same_registrable_domain:
            if (s_dom, s_suf) != (u_dom, u_suf):
                return False
            if include_subdomains:
                return True
            s_host = f"{s_dom}.{s_suf}" if not s_sub else f"{s_sub}.{s_dom}.{s_suf}"
            u_host = f"{u_dom}.{u_suf}" if not u_sub else f"{u_sub}.{u_dom}.{u_suf}"
            return s_host == u_host
        return urlsplit(url).netloc.lower() == urlsplit(site_root).netloc.lower()
    except Exception:
        return False


def derive_site_slug(site_url: str) -> str:
    netloc = urlsplit(site_url).netloc or site_url
    _, dom, suf = get_registrable_domain(site_url)
    base = f"{dom}.{suf}" if dom and suf else netloc
    slug = file_safe_slug(base, maxlen=80)
    return slug or sha1_short(site_url)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- Logging ------------------------------------- #


def setup_root_logger(results_root: Path) -> None:
    log_path = results_root.parent / "scraper.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(site)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger("scraper")
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)


def get_site_logger(site_dir: Path, site_slug: str) -> logging.LoggerAdapter:
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(site)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(f"scraper.{site_slug}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(site_dir / "scrape.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = True
    return logging.LoggerAdapter(logger, extra={"site": site_slug})


# ------------------------ Rate Limiter & Retries --------------------------- #


class RateLimiter:
    def __init__(self, delay: float):
        self.delay = max(0.0, delay)
        self._lock = asyncio.Lock()
        self._last_time: float = 0.0

    async def wait(self) -> None:
        jitter = random.uniform(0, self.delay * 0.3) if self.delay > 0 else 0.0
        min_interval = self.delay + jitter
        async with self._lock:
            now = time.monotonic()
            wait_for = self._last_time + min_interval - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_time = time.monotonic()


async def fetch_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    logger: logging.LoggerAdapter,
    rate_limiter: RateLimiter,
    max_attempts: int = 4,
    backoff_factor: float = 0.8,
    timeout: int = 20,
) -> Optional[httpx.Response]:
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            await rate_limiter.wait()
            resp = await client.get(url, timeout=timeout, follow_redirects=True)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"Server error {resp.status_code}", request=resp.request, response=resp
                )
            return resp
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.HTTPStatusError) as e:
            wait = (2 ** (attempt - 1)) * backoff_factor
            logger.warning(f"Attempt {attempt} failed for {url}: {e}. Retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            logger.exception(f"Non-retriable error for {url}: {e}")
            return None
    logger.error(f"Exceeded retry limit for {url}")
    return None


# ------------------------------ Robots & Sitemaps -------------------------- #


SITEMAP_RE = re.compile(r"(?im)^\s*sitemap:\s*(?P<url>\S+)\s*$")


async def fetch_robots(
    client: httpx.AsyncClient, base_url: str, logger: logging.LoggerAdapter, rate_limiter: RateLimiter, timeout: int
) -> Optional[str]:
    """Fetch robots.txt text for the site's host, if available. Accept any text/* content-type."""
    parts = urlsplit(base_url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    resp = await fetch_with_retries(client, robots_url, logger=logger, rate_limiter=rate_limiter, timeout=timeout)
    if resp and resp.status_code == 200:
        ctype = resp.headers.get("Content-Type", "").lower()
        if "text" in ctype or ctype == "" or "plain" in ctype or "html" in ctype:
            try:
                return resp.text
            except UnicodeDecodeError:
                return resp.content.decode("utf-8", errors="replace")
    return None


def _strip_comment(line: str) -> str:
    """Strip inline comments (#...) while preserving URLs with # in query very rarely used."""
    # Per robots.txt conventions, # starts a comment unless inside a URL; keep it simple:
    i = line.find("#")
    return line if i == -1 else line[:i]


def _ua_token(config_user_agent: str) -> str:
    """Return the primary UA token (first product token) to match in robots groups."""
    # e.g., "MyBot/1.0 (+url)" -> "mybot"
    token = config_user_agent.split("/", 1)[0].strip().lower()
    return token or "*"


def _parse_allow_all_from_robots(robots_text: str, config_user_agent: str) -> bool:
    """
    Very lightweight robots parser to detect if a file explicitly allows all paths
    for our UA token or '*':

    User-agent: *
    Allow: /

    If such a rule exists in the matching UA group(s), treat as allow-all.
    """
    if not robots_text:
        return False

    ua_token = _ua_token(config_user_agent)
    groups: list[list[str]] = []
    current: list[str] = []

    # Normalize lines
    for raw in robots_text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        m = re.match(r"(?i)user-agent\s*:\s*(.+)$", line)
        if m:
            # start of a (new) group
            value = m.group(1).strip().lower()
            if current:
                groups.append(current)
                current = []
            current = [f"user-agent: {value}"]
            continue
        else:
            # directive line
            if current:
                current.append(line)
            else:
                # orphan directive before any UA: attach to implicit group
                current = [f"user-agent: *", line]
    if current:
        groups.append(current)

    # collect groups matching our UA or *
    matched_groups: list[list[str]] = []
    for g in groups:
        uas = [re.sub(r"(?i)user-agent\s*:\s*", "", ln).strip().lower() for ln in g if ln.lower().startswith("user-agent")]
        if ua_token in uas or "*" in uas:
            matched_groups.append(g)

    # check if any matching group contains Allow: /
    for g in matched_groups:
        for ln in g:
            if ln.lower().startswith("allow"):
                # Allow: / or Allow: /
                m2 = re.match(r"(?i)allow\s*:\s*(.*)$", ln)
                if m2 and m2.group(1).strip() == "/":
                    return True
    return False


def build_robot_rules(robots_text: Optional[str], site_url: str) -> robotparser.RobotFileParser:
    rp = robotparser.RobotFileParser()
    parts = urlsplit(site_url)
    rp.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
    if robots_text:
        rp.parse(robots_text.splitlines())
    return rp


async def parse_sitemaps_from_robots(
    client: httpx.AsyncClient,
    robots_text: Optional[str],
    *,
    logger: logging.LoggerAdapter,
    rate_limiter: RateLimiter,
    timeout: int,
) -> list[str]:
    if not robots_text:
        return []
    sitemap_urls = [m.group("url") for m in SITEMAP_RE.finditer(robots_text)]
    discovered: list[str] = []

    for sm_url in sitemap_urls:
        resp = await fetch_with_retries(client, sm_url, logger=logger, rate_limiter=rate_limiter, timeout=timeout)
        if not resp or resp.status_code != 200:
            logger.warning(f"Failed to fetch sitemap: {sm_url}")
            continue
        data = resp.content
        if sm_url.lower().endswith(".gz") or resp.headers.get("Content-Type", "").lower().startswith("application/x-gzip"):
            with contextlib.suppress(Exception):
                data = gzip.decompress(data)
        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(data)
            def tag_endswith(el, name: str) -> bool:
                return el.tag.lower().endswith(name)

            if tag_endswith(root, "sitemapindex"):
                for sm in root:
                    if tag_endswith(sm, "sitemap"):
                        for child in sm:
                            if tag_endswith(child, "loc") and child.text:
                                discovered.append(child.text.strip())
            elif tag_endswith(root, "urlset"):
                for url_el in root:
                    if tag_endswith(url_el, "url"):
                        for child in url_el:
                            if tag_endswith(child, "loc") and child.text:
                                discovered.append(child.text.strip())
            else:
                for loc in root.iter():
                    if tag_endswith(loc, "loc") and (loc.text or "").strip():
                        discovered.append(loc.text.strip())
        except Exception as e:
            logger.warning(f"Could not parse sitemap {sm_url}: {e}")
            continue

    return discovered


# ------------------------------- HTML Parsing ------------------------------- #


BOILERPLATE_TAGS = {"script", "style", "noscript", "svg", "footer", "nav", "template"}

ASSET_EXTENSIONS = re.compile(
    r"\.(?:png|jpe?g|gif|bmp|webp|ico|svg|mp4|mp3|wav|avi|mov|mkv|css|js|woff2?|ttf|eot|zip|rar|7z|tar|gz|tgz|bz2)$",
    re.I,
)


def extract_text_from_html(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in BOILERPLATE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    for el in soup.select('[role="navigation"], header, footer, nav'):
        el.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n", strip=True)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    body = "\n".join(lines)
    return f"{title}\n\n{body}" if title else body


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        href = normalize_url(href, base=base_url)
        scheme = urlsplit(href).scheme
        if scheme not in ("http", "https"):
            continue
        if ASSET_EXTENSIONS.search(href.split("?", 1)[0]):
            continue
        links.append(href)
    return links


# ------------------------------ Site Crawler ------------------------------- #


@dataclasses.dataclass
class RobotsEntry:
    parser: robotparser.RobotFileParser
    raw_text: Optional[str]
    allow_all: bool  # True if a matching UA group contains "Allow: /"


class SiteCrawler:
    """Per-site asynchronous crawler."""

    def __init__(self, site_url: str, cfg: Config) -> None:
        self.site_url = normalize_url(site_url)
        self.cfg = cfg
        self.site_slug = derive_site_slug(self.site_url)

        self.site_dir = Path(cfg.results_dir) / self.site_slug
        self.pages_dir = self.site_dir / "pages"
        self.files_pdf_dir = self.site_dir / "files" / "pdf"
        self.files_word_dir = self.site_dir / "files" / "word"
        for p in (self.site_dir, self.pages_dir, self.files_pdf_dir, self.files_word_dir):
            ensure_dir(p)

        self.logger = get_site_logger(self.site_dir, self.site_slug)
        self.rate_limiter = RateLimiter(cfg.delay)
        self.sem = asyncio.Semaphore(cfg.concurrency)

        self.queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        self.visited: set[str] = set()
        self.external_urls: set[str] = set()
        self.pages_saved: int = 0

        self.visited_path = self.site_dir / "visited.txt"
        self.external_path = self.site_dir / "external_urls.txt"

        # Cache robots per *host*
        self._robots_cache: dict[str, RobotsEntry] = {}

    # --------------------------- Public API -------------------------------- #

    async def run(self) -> None:
        self.logger.info(f"Starting crawl: {self.site_url}")
        self._load_visited()

        headers = {
            "User-Agent": self.cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en;q=0.7, *;q=0.5",
        }
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=max(self.cfg.concurrency, 10))
        timeout = httpx.Timeout(self.cfg.timeout)

        async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout, follow_redirects=True) as client:
            # Seed homepage
            await self._enqueue(self.site_url, depth=0)

            # Seed sitemaps for start host
            robots_text = None
            if self.cfg.respect_robots:
                try:
                    robots_text = await fetch_robots(client, self.site_url, self.logger, self.rate_limiter, self.cfg.timeout)
                except Exception as e:
                    self.logger.warning(f"Could not fetch robots.txt: {e}")
                try:
                    start_host = urlsplit(self.site_url).netloc
                    rp = build_robot_rules(robots_text, self.site_url)
                    allow_all = _parse_allow_all_from_robots(robots_text or "", self.cfg.user_agent)
                    self._robots_cache[start_host] = RobotsEntry(parser=rp, raw_text=robots_text, allow_all=allow_all)
                    if allow_all:
                        self.logger.info(f"Robots for {start_host} contains 'Allow: /' → treating as ALLOW-ALL")
                except Exception:
                    pass

            try:
                sitemap_urls = await parse_sitemaps_from_robots(
                    client, robots_text, logger=self.logger, rate_limiter=self.rate_limiter, timeout=self.cfg.timeout
                )
                for sm_url in sitemap_urls:
                    n = normalize_url(sm_url)
                    if in_same_scope(n, self.site_url, self.cfg.include_subdomains, self.cfg.same_registrable_domain):
                        await self._enqueue(n, depth=0)
            except Exception as e:
                self.logger.warning(f"Failed to parse sitemaps: {e}")

            workers = [asyncio.create_task(self._worker(client)) for _ in range(self.cfg.concurrency)]
            await self.queue.join()
            for w in workers:
                w.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*workers)

        self._write_external_urls()
        self._flush_visited()
        self.logger.info(f"Completed: {self.pages_saved} page(s) saved. External URLs: {len(self.external_urls)}")

    # --------------------------- Internal ---------------------------------- #

    def _load_visited(self) -> None:
        if self.visited_path.exists():
            try:
                content = self.visited_path.read_text(encoding="utf-8")
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        self.visited.add(line)
                self.logger.info(f"Loaded {len(self.visited)} visited URL(s) from visited.txt")
            except Exception as e:
                self.logger.warning(f"Could not read visited.txt: {e}")

    def _flush_visited(self) -> None:
        try:
            data = "\n".join(sorted(self.visited))
            self.visited_path.write_text(data, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"Failed writing visited.txt: {e}")

    def _write_external_urls(self) -> None:
        try:
            lines = "\n".join(sorted(self.external_urls))
            self.external_path.write_text(lines, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"Failed writing external_urls.txt: {e}")

    async def _enqueue(self, url: str, depth: int) -> None:
        url = normalize_url(url)
        if url in self.visited:
            return
        if depth > self.cfg.max_depth:
            return
        await self.queue.put((url, depth))

    async def _worker(self, client: httpx.AsyncClient) -> None:
        while True:
            url, depth = await self.queue.get()
            try:
                await self._process_url(client, url, depth)
            except Exception as e:
                self.logger.exception(f"Unhandled error processing {url}: {e}")
            finally:
                self.queue.task_done()

    async def _get_robot_entry_for(self, client: httpx.AsyncClient, url: str) -> Optional[RobotsEntry]:
        netloc = urlsplit(url).netloc
        if not netloc:
            return None
        entry = self._robots_cache.get(netloc)
        if entry:
            return entry
        robots_text = None
        try:
            robots_text = await fetch_robots(client, f"https://{netloc}", self.logger, self.rate_limiter, self.cfg.timeout)
        except Exception as e:
            self.logger.warning(f"Could not fetch robots for host {netloc}: {e}")
        rp = build_robot_rules(robots_text, f"https://{netloc}")
        allow_all = _parse_allow_all_from_robots(robots_text or "", self.cfg.user_agent)
        entry = RobotsEntry(parser=rp, raw_text=robots_text, allow_all=allow_all)
        self._robots_cache[netloc] = entry
        if allow_all:
            self.logger.info(f"Robots for {netloc} contains 'Allow: /' → treating as ALLOW-ALL")
        return entry

    def _robots_allow(self, entry: RobotsEntry, url: str) -> bool:
        """
        Decide if robots permits `url`:

        1) If parsed text clearly states Allow: / for our UA or '*', return True.
        2) Else, try parser.can_fetch(config UA), then parser.can_fetch('*').
        3) If parser errors or both return False, default to allow to avoid false negatives.
        """
        if entry.allow_all:
            return True
        rp = entry.parser
        try:
            if rp.can_fetch(self.cfg.user_agent, url):
                return True
            if rp.can_fetch("*", url):
                return True
            # If both deny, be conservative and return False.
            return False
        except Exception:
            # On any parser failure, be permissive (polite default).
            return True

    async def _process_url(self, client: httpx.AsyncClient, url: str, depth: int) -> None:
        if url in self.visited:
            return
        self.visited.add(url)

        if self.cfg.respect_robots:
            entry = await self._get_robot_entry_for(client, url)
            if entry and not self._robots_allow(entry, url):
                self.logger.info(f"Disallowed by robots.txt: {url}")
                return

        if not in_same_scope(url, self.site_url, self.cfg.include_subdomains, self.cfg.same_registrable_domain):
            self.external_urls.add(url)
            return

        if self.pages_saved >= self.cfg.max_pages:
            return

        async with self.sem:
            resp = await fetch_with_retries(
                client, url, logger=self.logger, rate_limiter=self.rate_limiter, timeout=self.cfg.timeout
            )
        if not resp:
            return

        final_url = normalize_url(str(resp.url))
        self.visited.add(final_url)

        if self.cfg.respect_robots and final_url != url:
            entry2 = await self._get_robot_entry_for(client, final_url)
            if entry2 and not self._robots_allow(entry2, final_url):
                self.logger.info(f"Disallowed by robots.txt after redirect: {final_url}")
                return

        content_type = resp.headers.get("Content-Type", "").lower()
        kind = detect_doc_kind(final_url, content_type)
        if kind in {"pdf", "doc", "docx"}:
            await self._download_doc(resp, final_url, kind)
            return
        if not is_probably_html(final_url, content_type):
            return

        try:
            html = resp.text
        except UnicodeDecodeError:
            html = resp.content.decode("utf-8", errors="replace")

        text = extract_text_from_html(html, base_url=final_url)
        self._save_page_text(final_url, text)
        self.pages_saved += 1

        if self.pages_saved % 25 == 0:
            self._flush_visited()

        try:
            for link in extract_links(html, base_url=final_url):
                n = normalize_url(link)
                if in_same_scope(n, self.site_url, self.cfg.include_subdomains, self.cfg.same_registrable_domain):
                    if n not in self.visited and depth + 1 <= self.cfg.max_depth and self.pages_saved < self.cfg.max_pages:
                        await self._enqueue(n, depth + 1)
                else:
                    self.external_urls.add(n)
        except Exception as e:
            self.logger.warning(f"Failed extracting links from {final_url}: {e}")

    # ------------------------------ I/O Helpers ----------------------------- #

    def _page_filename_from_url(self, url: str) -> str:
        parts = urlsplit(url)
        base = parts.path.strip("/") or "index"
        if parts.query:
            base = f"{base}-{sha1_short(parts.query)}"
        slug = file_safe_slug(base, maxlen=90)
        return f"{slug}-{sha1_short(url)}.txt"

    def _doc_filename_from_response(self, url: str, kind: str, resp: httpx.Response) -> str:
        cd = resp.headers.get("Content-Disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.I)
        if m:
            candidate = Path(m.group(1)).name
        else:
            candidate = Path(urlsplit(url).path).name
        if not candidate.lower().endswith(f".{kind}"):
            candidate = f"{Path(candidate).stem}.{kind}"
        safe = file_safe_slug(Path(candidate).stem, maxlen=90) or kind
        return f"{safe}-{sha1_short(url)}.{kind}"

    def _save_page_text(self, url: str, text: str) -> None:
        try:
            path = self.pages_dir / self._page_filename_from_url(url)
            path.write_text(text, encoding="utf-8")
            self.logger.info(f"Saved page: {url} -> {path.name}")
        except Exception as e:
            self.logger.exception(f"Failed saving page text for {url}: {e}")

    async def _download_doc(self, resp: httpx.Response, url: str, kind: str) -> None:
        try:
            out_dir = self.files_pdf_dir if kind == "pdf" else self.files_word_dir
            path = out_dir / self._doc_filename_from_response(url, kind, resp)
            with path.open("wb") as f:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        f.write(chunk)
            self.logger.info(f"Downloaded {kind.upper()}: {url} -> {path.name}")
        except Exception as e:
            self.logger.exception(f"Failed downloading {kind.upper()} from {url}: {e}")


# ------------------------------- CLI --------------------------------------- #


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async website scraper & crawler.")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to YAML configuration file.")
    return parser.parse_args(argv)


async def run_for_site(site_url: str, cfg: Config, root_results: Path) -> None:
    crawler = SiteCrawler(site_url, cfg)
    await crawler.run()


async def main_async(cfg: Config) -> None:
    results_root = Path(cfg.results_dir)
    ensure_dir(results_root)
    setup_root_logger(results_root)

    root_logger = logging.getLogger("scraper")
    root_adapter = logging.LoggerAdapter(root_logger, extra={"site": "ALL"})
    if not cfg.sites:
        root_adapter.error("No sites provided in config 'sites:'")
        return

    root_adapter.info(f"Starting scrape for {len(cfg.sites)} site(s)")
    for site in cfg.sites:
        try:
            await run_for_site(site, cfg, results_root)
        except KeyboardInterrupt:
            root_adapter.warning("Interrupted by user")
            break
        except Exception as e:
            root_adapter.exception(f"Error scraping {site}: {e}")
    root_adapter.info("All done.")


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    try:
        asyncio.run(main_async(cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()