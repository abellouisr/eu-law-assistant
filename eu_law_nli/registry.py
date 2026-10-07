"""The document registry: one JSON file per legal act in ``documents/``.

Adding a directive means adding a file there and running the ingest command.
Adding an amendment means adding a version entry to the act's file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config


@dataclass
class Version:
    id: str
    celex: str = ""  # EUR-Lex documents
    kind: str = "consolidated"  # "original" | "consolidated" | "document"
    date: str = ""  # ISO date the version applies from
    note: str = ""
    url: str = ""  # "web" documents: the page to read
    path: str = ""  # "web" documents: or a local file, relative to the project
    format: str = ""  # "web" documents: html | markdown | text (default: from the name)


@dataclass
class Document:
    id: str
    short_name: str
    title: str
    citation: str
    scope: str
    source: str = "eurlex"  # see sources.py
    reference_language: str = "EN"
    languages: list[str] = field(default_factory=list)  # "web" documents; EUR-Lex has all 24
    active_version: str = "latest"
    # Consolidated texts omit recitals; when true they are taken from the
    # original Official Journal text and quoted with links to it.
    recitals_from_original: bool = False
    # Two example questions for the start page, in English (translated on use).
    examples: list[str] = field(default_factory=list)
    versions: list[Version] = field(default_factory=list)

    def version(self, version_id: str | None = None) -> Version:
        wanted = version_id or self.active_version
        if wanted == "latest":
            return max(self.versions, key=lambda v: v.date)
        for v in self.versions:
            if v.id == wanted:
                return v
        raise KeyError(f"{self.id}: unknown version '{wanted}'")

    @property
    def original(self) -> Version | None:
        return next((v for v in self.versions if v.kind == "original"), None)


def load_document(doc_id: str, directory: Path | None = None) -> Document:
    path = (directory or config.DOCUMENTS_DIR) / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No registry entry for '{doc_id}' at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    versions = [Version(**v) for v in data.pop("versions", [])]
    return Document(versions=versions, **data)


def add_version(doc_id: str, version: Version, directory: Path | None = None) -> None:
    """Append a version to the act's registry file, keeping everything else."""
    path = (directory or config.DOCUMENTS_DIR) / f"{doc_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(v.get("celex") == version.celex for v in data.get("versions", [])):
        return
    data.setdefault("versions", []).append(asdict(version))
    config.write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def list_documents(directory: Path | None = None) -> list[str]:
    directory = directory or config.DOCUMENTS_DIR
    return sorted(p.stem for p in directory.glob("*.json"))
