"""Download a document from EUR-Lex and build its corpus.

    python -m eu_law_nli.ingest eecc                 # active version, English
    python -m eu_law_nli.ingest eecc --lang EN ET FR  # more official languages
    python -m eu_law_nli.ingest eecc --lang ALL       # all 24 official languages
    python -m eu_law_nli.ingest eecc --check-updates  # newer consolidated texts?
    python -m eu_law_nli.ingest eecc --sync           # check the live text, adopt updates
    python -m eu_law_nli.ingest eecc --from-file page.html --celex 32018L1972

Downloaded pages are kept in data/raw/ so a rebuild does not hit EUR-Lex again.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .corpus import Corpus
from .parser import parse_html
from .registry import Document, Version, load_document

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 eu-law-nli-prototype"
)


# The 24 official EU languages, with the ISO 639-2 codes the Publications
# Office repository expects.
OFFICIAL_LANGUAGES = {
    "BG": "bul", "CS": "ces", "DA": "dan", "DE": "deu", "EL": "ell", "EN": "eng",
    "ES": "spa", "ET": "est", "FI": "fin", "FR": "fra", "GA": "gle", "HR": "hrv",
    "HU": "hun", "IT": "ita", "LT": "lit", "LV": "lav", "MT": "mlt", "NL": "nld",
    "PL": "pol", "PT": "por", "RO": "ron", "SK": "slk", "SL": "slv", "SV": "swe",
}
CELLAR_BASE = "http://publications.europa.eu/resource/celex"
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"


class FetchError(RuntimeError):
    pass


def raw_path(celex: str, lang: str) -> Path:
    return config.RAW_DIR / f"{celex}_{lang.upper()}.html"


def _download(url: str, headers: dict | None = None) -> str:
    import requests

    resp = requests.get(url, headers={"User-Agent": USER_AGENT, **(headers or {})}, timeout=120)
    if resp.status_code != 200 or len(resp.text) < 20000:
        raise FetchError(
            f"EUR-Lex returned HTTP {resp.status_code} ({len(resp.text)} characters) for {url}"
        )
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def fetch_html(celex: str, lang: str, refresh: bool = False) -> str:
    """Return the page for one CELEX number and language, using the cache first."""
    path = raw_path(celex, lang)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    url = config.eurlex_url(celex, lang)
    try:
        try:
            html = _download(url)
        except Exception:
            # EUR-Lex often turns away scripts; the Publications Office
            # repository serves the same XHTML and allows automated access.
            html = _download_cellar(celex, lang)
    except Exception as exc:  # network error, automated-traffic check, wrong CELEX
        raise FetchError(
            f"Could not download {celex} ({lang.upper()}) automatically: {exc}\n"
            "EUR-Lex often turns away automated downloads (HTTP 202 with an empty page).\n"
            f"Open this page in a browser:\n  {url}\n"
            f"save it as 'Web page, HTML only' to:\n  {path}\n"
            "and run the same ingest command again."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


def download_live(celex: str, lang: str) -> str:
    """The current text straight from the EU servers, bypassing data/raw/."""
    return _download_cellar(celex, lang)


def _download_cellar(celex: str, lang: str) -> str:
    code = OFFICIAL_LANGUAGES.get(lang.upper())
    if code is None:
        raise FetchError(f"{lang} is not an official EU language")
    return _download(f"{CELLAR_BASE}/{celex}",
                     {"Accept": "application/xhtml+xml", "Accept-Language": code})


def build_corpus(doc: Document, version: Version, lang: str, refresh: bool = False) -> Corpus:
    """Fetch, parse and store one version in one language, whatever its source."""
    from .sources import for_document
    return for_document(doc).build(doc, version, lang, refresh)


def build_eurlex_corpus(doc: Document, version: Version, lang: str,
                        refresh: bool = False) -> Corpus:
    """Parse one EUR-Lex version. When the registry switches on
    ``recitals_from_original``, recitals are taken from the original act if the
    chosen version (a consolidated text) does not carry them."""
    html = fetch_html(version.celex, lang, refresh)
    provisions = parse_html(html, source_celex=version.celex)
    if not any(p.kind == "article" for p in provisions):
        raise FetchError(
            f"No articles found in {version.celex} ({lang}). The saved page is "
            "probably not the EUR-Lex HTML text of the act."
        )
    sources = {version.celex: hashlib.sha256(html.encode("utf-8")).hexdigest()}
    original = doc.original
    if doc.recitals_from_original and original and original.celex != version.celex and not any(
        p.kind == "recital" for p in provisions
    ):
        original_html = fetch_html(original.celex, lang, refresh)
        recitals = [p for p in parse_html(original_html, original.celex) if p.kind == "recital"]
        provisions = recitals + provisions
        sources[original.celex] = hashlib.sha256(original_html.encode("utf-8")).hexdigest()

    counts = {k: sum(1 for p in provisions if p.kind == k) for k in ("article", "recital", "annex")}
    corpus = Corpus(
        document=doc.id, version=version.id, lang=lang.upper(), provisions=provisions,
        meta={
            "source": "eurlex", "celex": version.celex, "kind": version.kind, "date": version.date,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources_sha256": sources, "counts": counts,
        },
    )
    corpus.save()
    return corpus


def discover_consolidated(celex: str) -> list[str]:
    """CELEX numbers of the consolidated versions published for an act, oldest
    first. Asks the Publications Office database behind EUR-Lex (its public
    SPARQL service), because EUR-Lex's own pages turn scripts away."""
    import requests

    stem = "0" + celex[1:] + "-"
    query = (
        "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>\n"
        "SELECT DISTINCT ?celex WHERE { ?act cdm:resource_legal_id_celex ?celex . "
        f'FILTER(STRSTARTS(STR(?celex), "{stem}")) }}'
    )
    resp = requests.get(
        SPARQL_ENDPOINT, timeout=90, headers={"User-Agent": USER_AGENT},
        params={"query": query, "format": "application/sparql-results+json"},
    )
    resp.raise_for_status()
    found = {b["celex"]["value"] for b in resp.json()["results"]["bindings"]}
    return sorted(c for c in found if re.fullmatch(re.escape(stem) + r"\d{8}", c))


def document_date(celex: str) -> str:
    """The date of an act (its adoption date), from the Publications Office."""
    import requests

    query = (
        "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>\n"
        "SELECT ?date WHERE { ?work cdm:resource_legal_id_celex "
        f'"{celex}"^^<http://www.w3.org/2001/XMLSchema#string> ; '
        "cdm:work_date_document ?date }"
    )
    resp = requests.get(
        SPARQL_ENDPOINT, timeout=90, headers={"User-Agent": USER_AGENT},
        params={"query": query, "format": "application/sparql-results+json"},
    )
    resp.raise_for_status()
    rows = resp.json()["results"]["bindings"]
    return rows[0]["date"]["value"][:10] if rows else ""


def version_date(celex: str) -> str:
    """ISO date of a consolidated version: 02018L1972-20241018 -> 2024-10-18."""
    stamp = celex[-8:]
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


def add_eurlex_document(doc_id: str, celex: str, short_name: str, citation: str,
                        scope: str, title: str | None = None) -> Path:
    """Write a registry file for an EU act: the Official Journal text plus the
    latest consolidated version published, if any."""
    import json
    from dataclasses import asdict

    path = config.DOCUMENTS_DIR / f"{doc_id}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists already")
    try:
        adopted = document_date(celex)
    except Exception:  # the service is slow or down: the date is not essential
        adopted = ""
    original = Version(id="original", celex=celex, kind="original", date=adopted,
                       note=f"Text as adopted on {adopted} and published in the Official "
                            "Journal." if adopted else "Text as published in the Official Journal.")
    versions = [original]
    consolidated = discover_consolidated(celex)
    if consolidated:
        latest = consolidated[-1]
        versions.append(Version(id=version_date(latest), celex=latest, kind="consolidated",
                                date=version_date(latest),
                                note="Latest consolidated text on EUR-Lex when added."))
    data ={"id": doc_id, "short_name": short_name, "title": title or citation,
            "citation": citation, "scope": scope, "source": "eurlex",
            "reference_language": "EN", "active_version": "latest",
            "recitals_from_original": False, "examples": [],
            "versions": [asdict(v) for v in versions]}
    config.write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return path


def check_updates(doc: Document) -> list[str]:
    original = doc.original
    if original is None:
        return []
    known = {v.celex for v in doc.versions}
    current = doc.version("latest").date
    return [c for c in discover_consolidated(original.celex)
            if c not in known and version_date(c) > current]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("document", help="registry id, e.g. eecc")
    ap.add_argument("--lang", nargs="+", default=None, help="language codes, e.g. EN ET FR")
    ap.add_argument("--version", default=None, help="version id from the registry (default: active)")
    ap.add_argument("--refresh", action="store_true", help="download again even if cached")
    ap.add_argument("--check-updates", action="store_true", help="list consolidated versions not yet in the registry")
    ap.add_argument("--sync", action="store_true",
                    help="check the live EU text now and adopt a newer version if there is one")
    ap.add_argument("--from-file", type=Path, help="use a page saved from the browser")
    ap.add_argument("--add-eurlex", metavar="CELEX",
                    help="create documents/<document>.json for this EU act, then ingest it; "
                         "needs --short-name, --citation and --scope")
    ap.add_argument("--short-name", help="with --add-eurlex: e.g. 'General Data Protection Regulation'")
    ap.add_argument("--citation", help="with --add-eurlex: e.g. 'Regulation (EU) 2016/679'")
    ap.add_argument("--scope", help="with --add-eurlex: one or two sentences on what the act covers")
    ap.add_argument("--title", help="with --add-eurlex: full title (default: the citation)")
    ap.add_argument("--celex", help="CELEX number the saved page belongs to (with --from-file)")
    args = ap.parse_args(argv)

    if args.add_eurlex:
        if not (args.short_name and args.citation and args.scope):
            ap.error("--add-eurlex needs --short-name, --citation and --scope")
        path = add_eurlex_document(args.document, args.add_eurlex, args.short_name,
                                   args.citation, args.scope, args.title)
        print(f"Created {path}")
    doc = load_document(args.document)
    if args.sync:
        from .live import ensure_current

        result = ensure_current(doc.id, force=True)
        if result["ok"]:
            print(f"[{doc.id}] {result['action']}: version {result['version']} "
                  f"({result['celex']}), checked {result['checked_at']}")
        return 0 if result["ok"] else 1
    if args.check_updates:
        try:
            new = check_updates(doc)
        except Exception as exc:
            print(f"Could not check the EU Publications Office: {exc}", file=sys.stderr)
            return 1
        if not new:
            print("No consolidated versions beyond those in the registry.")
        for celex in new:
            print(f"New consolidated version: {celex} (applies from {version_date(celex)}). "
                  f"Run: python -m eu_law_nli.ingest {doc.id} --sync")
        return 0

    langs = [l.upper() for l in (args.lang or [doc.reference_language])]
    if langs == ["ALL"]:
        from .sources import for_document
        langs = for_document(doc).languages(doc)
    version = doc.version(args.version)

    if args.from_file:
        celex = args.celex or version.celex
        if len(langs) != 1:
            ap.error("--from-file needs exactly one --lang")
        target = raw_path(celex, langs[0])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.from_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        print(f"Stored {args.from_file} as {target}")

    failed = []
    for lang in langs:
        try:
            corpus = build_corpus(doc, version, lang, refresh=args.refresh)
        except (FetchError, FileNotFoundError) as exc:
            print(f"[{lang}] {exc}", file=sys.stderr)
            failed.append(lang)
            continue
        counts = ", ".join(f"{n} {kind}{'es' if kind.endswith('x') else 's'}" for kind, n in corpus.meta["counts"].items() if n)
        print(f"[{lang}] {doc.citation}, version {version.id}: {counts} -> "
              f"{Corpus.path_for(doc.id, version.id, lang)}")
    if failed:
        print(f"Not built: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
