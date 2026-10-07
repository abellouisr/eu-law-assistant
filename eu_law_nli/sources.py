"""Where the text of a document comes from.

Each registry file in documents/ names a source type in "source". A source
knows how to fetch and parse a version, link to a provision, and check
whether the live text has moved on. Everything else (search, answers,
quotation checks, the live check, the web app) works the same for every
source, so adding a kind of reference material means adding a class here.

    "eurlex"  EU acts by CELEX number, all official languages, consolidated
              versions discovered automatically (the default).
    "web"     any other reference point: a web page or a local file in HTML,
              Markdown or plain text (guidelines, national laws, internal
              notes). The text is split into sections at its headings. The
              live check downloads it again and rebuilds when it changed.

A "web" registry file looks like documents/templates/web-source.example.json.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .corpus import Corpus
from .parser import Block, Provision, normalise
from .registry import Document, Version


class SourceError(RuntimeError):
    pass


def for_document(doc: Document) -> "EurLexSource | WebSource":
    try:
        return SOURCES[doc.source]()
    except KeyError:
        raise SourceError(f"{doc.id}: unknown source type '{doc.source}' "
                          f"(known: {', '.join(SOURCES)})") from None


def link(doc: Document, version: Version, lang: str, provision: Provision | None = None) -> str:
    """Where a reader can see the official text of a provision (or the document)."""
    return for_document(doc).link(doc, version, lang, provision)


def _save(doc: Document, version: Version, lang: str, provisions: list[Provision],
          raw: str, ref: str) -> Corpus:
    counts: dict[str, int] = {}
    for p in provisions:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    corpus = Corpus(
        document=doc.id, version=version.id, lang=lang.upper(), provisions=provisions,
        meta={
            "source": doc.source, "ref": ref, "kind": version.kind, "date": version.date,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources_sha256": {ref: hashlib.sha256(raw.encode("utf-8")).hexdigest()},
            "counts": counts,
        },
    )
    corpus.save()
    return corpus


class EurLexSource:
    """EU acts from EUR-Lex and the Publications Office repository."""

    def build(self, doc: Document, version: Version, lang: str, refresh: bool = False) -> Corpus:
        from . import ingest
        return ingest.build_eurlex_corpus(doc, version, lang, refresh)

    def live_provisions(self, doc: Document, version: Version, lang: str) -> list[Provision]:
        from . import ingest
        from .parser import parse_html
        return parse_html(ingest.download_live(version.celex, lang), source_celex=version.celex)

    def newer_versions(self, doc: Document, current: Version) -> list[Version]:
        from . import ingest
        original = doc.original
        if original is None:
            return []
        known = {v.celex for v in doc.versions}
        out = []
        for celex in ingest.discover_consolidated(original.celex):
            date = ingest.version_date(celex)
            if celex not in known and date > current.date:
                out.append(Version(id=date, celex=celex, kind="consolidated", date=date))
        return out

    def link(self, doc: Document, version: Version, lang: str,
             provision: Provision | None = None) -> str:
        celex = (provision.source_celex if provision else "") or version.celex
        return config.eurlex_url(celex, lang, provision.id if provision else "")

    def languages(self, doc: Document) -> list[str]:
        from .ingest import OFFICIAL_LANGUAGES
        return list(OFFICIAL_LANGUAGES)


class WebSource:
    """A web page or a local file: HTML, Markdown or plain text."""

    def build(self, doc: Document, version: Version, lang: str, refresh: bool = False) -> Corpus:
        raw = self._read(doc, version, lang, refresh)
        provisions = self._parse(raw, version)
        if not provisions:
            raise SourceError(f"{doc.id}: no text found in {self._ref(version)}")
        return _save(doc, version, lang, provisions, raw, self._ref(version))

    def live_provisions(self, doc: Document, version: Version, lang: str) -> list[Provision]:
        return self._parse(self._read(doc, version, lang, refresh=True), version)

    def newer_versions(self, doc: Document, current: Version) -> list[Version]:
        return []  # a web page has one current version; changes are picked up as a refresh

    def link(self, doc: Document, version: Version, lang: str,
             provision: Provision | None = None) -> str:
        if not version.url:
            return ""
        anchor = provision.id if provision and not provision.id.startswith("sec_") else ""
        return f"{version.url}#{anchor}" if anchor else version.url

    def languages(self, doc: Document) -> list[str]:
        return [l.upper() for l in (doc.languages or [doc.reference_language])]

    # -- reading ------------------------------------------------------
    @staticmethod
    def _ref(version: Version) -> str:
        return version.url or version.path

    def _read(self, doc: Document, version: Version, lang: str, refresh: bool) -> str:
        if version.path:
            path = Path(version.path)
            if not path.is_absolute():
                path = config.ROOT / path
            if not path.exists():
                raise SourceError(f"{doc.id}: file not found: {path}")
            return path.read_text(encoding="utf-8", errors="replace")
        if not version.url:
            raise SourceError(f"{doc.id}: version {version.id} has neither 'url' nor 'path'")
        cache = config.RAW_DIR / f"{doc.id}_{version.id}_{lang.upper()}.{version.format or 'html'}"
        if cache.exists() and not refresh:
            return cache.read_text(encoding="utf-8")
        import requests
        from .ingest import USER_AGENT
        resp = requests.get(version.url, headers={"User-Agent": USER_AGENT}, timeout=120)
        if resp.status_code != 200:
            raise SourceError(f"{doc.id}: HTTP {resp.status_code} for {version.url}")
        resp.encoding = resp.encoding or "utf-8"
        config.write_text_atomic(cache, resp.text)
        return resp.text

    def _parse(self, raw: str, version: Version) -> list[Provision]:
        fmt = (version.format or Path(self._ref(version)).suffix.lstrip(".") or "html").lower()
        if fmt in ("md", "markdown"):
            return _sections_from_markdown(raw)
        if fmt in ("txt", "text"):
            return _sections_from_text(raw)
        return _sections_from_html(raw)


def _section(number: int, heading: str, path: list[str], paragraphs: list[str],
             anchor: str = "") -> Provision:
    return Provision(
        id=anchor or f"sec_{number}", kind="section", number=str(number),
        label=heading or f"Section {number}", path=list(path),
        blocks=[Block(text=t) for t in paragraphs if t.strip()],
    )


def _sections_from_markdown(raw: str) -> list[Provision]:
    sections, path, heading, paragraphs, buf = [], [], "", [], []

    def flush_paragraph() -> None:
        if buf:
            paragraphs.append(normalise(" ".join(buf)))
            buf.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append(_section(len(sections) + 1, heading, path[:-1], paragraphs[:]))
        paragraphs.clear()

    for line in raw.splitlines():
        match = re.match(r"^(#{1,4})\s+(.*)", line)
        if match:
            flush_section()
            level, heading = len(match.group(1)), match.group(2).strip()
            path[:] = path[:level - 1] + [heading]
        elif not line.strip():
            flush_paragraph()
        else:
            buf.append(line.strip())
    flush_section()
    return sections


def _sections_from_text(raw: str) -> list[Provision]:
    paragraphs = [normalise(p) for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return [_section(1, "", [], paragraphs)] if paragraphs else []


def _sections_from_html(raw: str) -> list[Provision]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    sections, path, heading, anchor, paragraphs = [], [], "", "", []

    def flush() -> None:
        if paragraphs:
            sections.append(_section(len(sections) + 1, heading, path[:-1], paragraphs[:],
                                     anchor))
        paragraphs.clear()

    for tag in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        if tag.name.startswith("h"):
            flush()
            level, heading = int(tag.name[1]), normalise(tag.get_text(" "))
            anchor = tag.get("id", "")
            path[:] = path[:level - 1] + [heading]
        elif not tag.find(["p", "li"]):  # leaf text only, so nothing is counted twice
            text = normalise(tag.get_text(" "))
            if text:
                paragraphs.append(text)
    flush()
    return sections


SOURCES = {"eurlex": EurLexSource, "web": WebSource}
