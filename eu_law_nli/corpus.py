"""A parsed version of a document in one language, stored as JSON on disk."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .parser import Provision, normalise

CHUNK_CHARS = 1600
HARD_LIMIT = 3200


@dataclass
class Chunk:
    """A retrievable slice of a provision: a run of consecutive blocks."""

    provision_id: str
    start: int  # index of the first block
    end: int  # index after the last block
    text: str


@dataclass
class Corpus:
    document: str
    version: str
    lang: str
    meta: dict = field(default_factory=dict)
    provisions: list[Provision] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id = {p.id: p for p in self.provisions}

    def get(self, provision_id: str) -> Provision | None:
        return self._by_id.get(provision_id)

    def chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        for p in self.provisions:
            out.extend(chunk_provision(p))
        return out

    def outline(self) -> str:
        """Compact table of contents, used to tell the model what is in scope."""
        lines: list[str] = []
        last_path: list[str] = []
        for p in self.provisions:
            if p.kind == "article":
                if p.path != last_path:
                    lines.append(" > ".join(p.path))
                    last_path = p.path
                lines.append(f"  {p.id}: {p.label} — {p.title}")
            elif p.kind in ("annex", "section"):
                lines.append(f"{p.id}: {p.label} — {p.title}")
        recitals = sum(1 for p in self.provisions if p.kind == "recital")
        if recitals:
            lines.append(f"Recitals rct_1 to rct_{recitals} (explanatory preamble)")
        return "\n".join(lines)

    # -- storage ---------------------------------------------------------
    @staticmethod
    def path_for(document: str, version: str, lang: str) -> Path:
        return config.CORPUS_DIR / document / version / f"{lang.upper()}.json"

    def save(self) -> Path:
        path = self.path_for(self.document, self.version, self.lang)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "document": self.document, "version": self.version, "lang": self.lang,
            "meta": self.meta, "provisions": [p.to_dict() for p in self.provisions],
        }
        config.write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=1))
        return path

    @classmethod
    def load(cls, document: str, version: str, lang: str) -> "Corpus":
        path = cls.path_for(document, version, lang)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            document=data["document"], version=data["version"], lang=data["lang"],
            meta=data.get("meta", {}),
            provisions=[Provision.from_dict(p) for p in data["provisions"]],
        )

    @classmethod
    def available_languages(cls, document: str, version: str) -> list[str]:
        folder = config.CORPUS_DIR / document / version
        return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []


def _split_long(text: str) -> list[str]:
    """Break one oversized block at sentence ends so no chunk is unbounded."""
    if len(text) <= HARD_LIMIT:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        while len(sentence) > HARD_LIMIT:  # e.g. a table with no punctuation
            cut = sentence.rfind(" ", 0, HARD_LIMIT)
            cut = cut if cut > 0 else HARD_LIMIT
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].strip()
        if current and len(current) + len(sentence) + 1 > CHUNK_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_provision(p: Provision) -> list[Chunk]:
    chunks: list[Chunk] = []
    start, buf = 0, ""
    for i, block in enumerate(p.blocks):
        if len(block.text) > HARD_LIMIT:
            if buf:
                chunks.append(Chunk(p.id, start, i, buf))
                buf = ""
            for piece in _split_long(block.text):
                chunks.append(Chunk(p.id, i, i + 1, piece))
            start = i + 1
            continue
        if buf and len(buf) + len(block.text) + 1 > CHUNK_CHARS:
            chunks.append(Chunk(p.id, start, i, buf))
            start, buf = i, ""
        buf = f"{buf}\n{block.text}".strip()
    if buf:
        chunks.append(Chunk(p.id, start, len(p.blocks), buf))
    return chunks


def reference(p: Provision, quote: str = "") -> str:
    """Human-readable pinpoint reference, e.g. "Article 61(2)" or "Recital 12".

    When ``quote`` sits inside one numbered paragraph, that paragraph number
    (and point, for definition lists) is added.
    """
    if p.kind == "recital":
        return f"Recital {p.number}"
    if not quote:
        return p.label
    needle = normalise(quote)
    hits = [b for b in p.blocks if needle in normalise(b.text)]
    if len(hits) != 1:
        return p.label
    block = hits[0]
    suffix = f"({block.para})" if block.para else ""
    if block.point:
        suffix += f"({block.point})"
    return f"{p.label}{suffix}"
