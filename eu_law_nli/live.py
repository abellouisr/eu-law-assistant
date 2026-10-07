"""Keep the stored text in line with the live text published by the EU.

``start_background`` runs the check in a background thread: once when the web
app or the terminal chat starts, then every config.LIVE_CHECK_HOURS, and on
demand through ``check_now``. Nobody waits for it; the assistant answers from
the stored copy meanwhile and switches to a newer text once it is built.
Each check (``sync``):

  1. asks the Publications Office database (the source of EUR-Lex) which
     consolidated versions of the act exist;
  2. if one is newer than the version in use, downloads it in every language
     already built, records it in documents/<id>.json and so makes it the
     version the assistant uses;
  3. otherwise downloads the reference-language text of the version in use
     and compares it with the stored copy, rebuilding all languages if the
     wording differs.

The outcome is saved in data/live/<id>.json and shown in the source line of
every reply. If the EU servers cannot be reached, the assistant keeps using
the stored copy and the source line says the check failed.

    python -m eu_law_nli.ingest eecc --sync     # run the same check by hand
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

from . import config
from .corpus import Corpus
from .registry import Version, add_version, load_document

_lock = threading.Lock()  # held for a whole check, which can take minutes
_checked_this_process: dict[str, datetime] = {}

# Separate from _lock, so the page never waits for a running check just to
# confirm that the background thread exists.
_thread_lock = threading.Lock()
_threads: dict[str, threading.Thread] = {}
_wake: dict[str, threading.Event] = {}
_stopping: set[str] = set()
_running: set[str] = set()


def start_background(doc_id: str, log=lambda message: None) -> None:
    """Start the background checker for an act, unless it is already running."""
    if not config.LIVE_CHECK:
        return
    with _thread_lock:
        thread = _threads.get(doc_id)
        if thread is not None and thread.is_alive():
            return
        wake = _wake.setdefault(doc_id, threading.Event())
        thread = threading.Thread(target=_loop, args=(doc_id, wake, log),
                                  name=f"live-check-{doc_id}", daemon=True)
        _threads[doc_id] = thread
        thread.start()


def check_now(doc_id: str) -> None:
    """Ask the background checker to check right away."""
    start_background(doc_id)
    if doc_id in _wake:
        _wake[doc_id].set()


def is_checking(doc_id: str) -> bool:
    return doc_id in _running


def stop_background(doc_id: str, timeout: float = 10) -> None:
    """Stop the checker (used by the tests)."""
    thread = _threads.get(doc_id)
    if thread is None:
        return
    _stopping.add(doc_id)
    _wake[doc_id].set()
    thread.join(timeout)
    _stopping.discard(doc_id)
    _threads.pop(doc_id, None)


def _loop(doc_id: str, wake: threading.Event, log) -> None:
    while doc_id not in _stopping:
        _running.add(doc_id)
        try:
            ensure_current(doc_id, force=True, log=log)
        except Exception as exc:  # keep the thread alive whatever happens
            log(f"[{doc_id}] Live check failed: {exc}")
        finally:
            _running.discard(doc_id)
        wake.wait(timeout=config.LIVE_CHECK_HOURS * 3600)
        wake.clear()


def text_stamp(doc_id: str, version_id: str, lang: str) -> float:
    """Changes whenever the stored text of a version is rebuilt, so callers
    that cache an engine know to load the new text."""
    path = Corpus.path_for(doc_id, version_id, lang)
    return path.stat().st_mtime if path.exists() else 0.0


def status_path(doc_id: str):
    return config.LIVE_DIR / f"{doc_id}.json"


def status(doc_id: str) -> dict:
    """The last check's outcome, or {} if the act has never been checked."""
    path = status_path(doc_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def ensure_current(doc_id: str, force: bool = False, log=print) -> dict:
    """Check once per process start and then at most every LIVE_CHECK_HOURS."""
    if not config.LIVE_CHECK and not force:
        return status(doc_id)
    with _lock:
        last = _checked_this_process.get(doc_id)
        due = last is None or datetime.now(timezone.utc) - last >= timedelta(
            hours=config.LIVE_CHECK_HOURS)
        if not (due or force):
            return status(doc_id)
        result = sync(doc_id, log=log)
        _checked_this_process[doc_id] = datetime.now(timezone.utc)
        return result


def sync(doc_id: str, log=print) -> dict:
    from . import ingest  # here to keep module loading light for the tests
    from .sources import for_document

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc = load_document(doc_id)
    source = for_document(doc)
    current = doc.version("latest")
    result = {"checked_at": now, "ok": False, "version": current.id,
              "celex": current.celex or current.url or current.path, "action": "", "error": "",
              "last_ok_at": status(doc_id).get("last_ok_at", "")}
    try:
        newer = source.newer_versions(doc, current)
        langs = Corpus.available_languages(doc.id, current.id) or [doc.reference_language]
        if newer:
            version = newer[-1]
            version.note = version.note or (
                f"Consolidated text adopted automatically from the EU Publications Office "
                f"on {now[:10]}. Consolidated texts are documentation tools and have no "
                "legal effect; only the Official Journal text is authentic.")
            log(f"[{doc.id}] Newer version published: {version.celex or version.id}. "
                f"Building {len(langs)} language(s)...")
            failed = _build(doc, version, langs, ingest, refresh=False, log=log)
            if doc.reference_language.upper() in failed:
                raise RuntimeError(f"could not build the {doc.reference_language} text of "
                                   f"{version.celex or version.id}")
            add_version(doc.id, version)
            result.update(version=version.id, celex=version.celex,
                          action=f"adopted {version.id}" + (
                              f" (not built: {', '.join(failed)})" if failed else ""))
        else:
            if _differs_from_live(doc, current, source):
                log(f"[{doc.id}] The live text of version {current.id} differs from the "
                    "stored copy. Rebuilding...")
                failed = _build(doc, current, langs, ingest, refresh=True, log=log)
                result["action"] = "refreshed" + (
                    f" (not built: {', '.join(failed)})" if failed else "")
            else:
                result["action"] = "up to date"
        result["ok"] = True
        result["last_ok_at"] = now
    except Exception as exc:  # offline, server error, unexpected page
        result["error"] = str(exc)[:300]
        log(f"[{doc.id}] Could not check the live text: {result['error']}. "
            "Using the stored copy.")
    config.write_text_atomic(status_path(doc.id), json.dumps(result, indent=2))
    return result


def _build(doc, version: Version, langs: list[str], ingest, refresh: bool, log) -> list[str]:
    failed = []
    for lang in langs:
        try:
            ingest.build_corpus(doc, version, lang, refresh=refresh)
        except Exception as exc:
            log(f"[{doc.id}] {lang}: {exc}")
            failed.append(lang.upper())
    return failed


def _differs_from_live(doc, version: Version, source) -> bool:
    """Compare the stored reference-language text with the live one, provision
    by provision, so formatting-only differences in the page do not count."""
    lang = doc.reference_language.upper()
    live = source.live_provisions(doc, version, lang)
    try:
        stored = Corpus.load(doc.id, version.id, lang)
    except FileNotFoundError:
        return True
    texts = lambda provisions: [(p.id, p.text) for p in provisions if p.kind != "recital"]
    return texts(live) != texts(stored.provisions)
