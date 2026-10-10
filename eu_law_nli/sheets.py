"""Copy of the usage and topics logs in a private Google Sheet.

A hosted copy (Streamlit Community Cloud) loses its local files whenever it
restarts, and its files cannot be opened. With config.GSHEET_ID set, every
usage entry is also added as a row to the sheet's "Usage" tab and every
recorded topic to its "Topics" tab, which only the sheet's owner (and people
they share it with) can see.

Rows are sent by one background thread, in order, so answers never wait for
Google; a failure is logged and never affects the reply. Rows older than
config.USAGE_RETENTION_DAYS are deleted from the sheet too, at most every few
hours, so the 90-day notice holds there as well.

Set-up: README, "Usage log in a Google Sheet".
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config

log = logging.getLogger("eu_law_nli")

USAGE_TAB = "Usage"
USAGE_COLUMNS = ["time", "session", "interface", "document", "version", "language", "kind",
                 "question", "articles", "quotes_kept", "quotes_dropped", "outside_topics",
                 "seconds", "model", "provider", "input_tokens", "output_tokens", "error"]
TOPICS_TAB = "Topics"
TOPICS_COLUMNS = ["received_at", "document", "version", "language", "question",
                  "question_en", "reason", "topics"]
SOURCES = {"national_law": "national law", "other_eu_law": "other EU act", "other": "other"}
PURGE_EVERY_SECONDS = 6 * 3600

_queue: "queue.Queue[tuple[str, list[str], dict]]" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
_spreadsheet = None  # opened once per process
_tabs: dict = {}
_last_purge: dict[str, float] = {}


def enabled() -> bool:
    return bool(config.GSHEET_ID)


def add_usage(entry: dict) -> None:
    _send(USAGE_TAB, USAGE_COLUMNS, entry)


def add_topic(entry: dict) -> None:
    _send(TOPICS_TAB, TOPICS_COLUMNS, entry)


def _send(tab: str, columns: list[str], entry: dict) -> None:
    """Queue one row for the background thread; returns at once."""
    global _worker
    if not enabled():
        return
    _queue.put((tab, columns, dict(entry)))
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="sheets-log", daemon=True)
            _worker.start()


def wait(timeout: float = 30) -> None:
    """Block until queued rows are sent (for tests and short scripts)."""
    deadline = time.monotonic() + timeout
    while _queue.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.05)


def _run() -> None:
    while True:
        tab, columns, entry = _queue.get()
        try:
            worksheet = _worksheet(tab, columns)
            worksheet.append_row([_cell(entry.get(c)) for c in columns],
                                 value_input_option="RAW")
            _purge_if_due(tab, worksheet)
        except Exception as exc:  # never let logging break the app
            log.warning("Could not write to the Google Sheet (%s): %s", tab, exc)
        finally:
            _queue.task_done()


def _cell(value) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        return ", ".join(_describe(v) for v in value)
    return str(value)


def _describe(item) -> str:
    if isinstance(item, dict):  # a topic: "label (source)"
        label = item.get("topic_en") or item.get("topic") or ""
        source = SOURCES.get(item.get("source", "other"), "other")
        return f"{label} ({source})" if label else ""
    return str(item)


def _open():
    """The spreadsheet, through a service account (gspread)."""
    import gspread  # imported here: only needed when the sheet is in use

    raw = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if raw:
        info = json.loads(raw)
    elif config.GSHEET_CREDENTIALS_FILE:
        path = Path(config.GSHEET_CREDENTIALS_FILE)
        path = path if path.is_absolute() else config.ROOT / path
        info = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError("no credentials: set GCP_SERVICE_ACCOUNT_JSON or "
                           "NLI_GSHEET_CREDENTIALS_FILE")
    return gspread.service_account_from_dict(info).open_by_key(config.GSHEET_ID)


def _worksheet(tab: str, columns: list[str]):
    """The tab, created with a header row the first time."""
    global _spreadsheet
    if tab in _tabs:
        return _tabs[tab]
    if _spreadsheet is None:
        _spreadsheet = _open()
    try:
        worksheet = _spreadsheet.worksheet(tab)
    except Exception:  # not there yet
        worksheet = _spreadsheet.add_worksheet(title=tab, rows=1000, cols=len(columns))
        worksheet.append_row(columns, value_input_option="RAW")
    _tabs[tab] = worksheet
    return worksheet


def _purge_if_due(tab: str, worksheet) -> None:
    """Delete rows older than the retention period. Rows are added in time
    order, so the expired ones are the first rows under the header."""
    now = time.monotonic()
    if now - _last_purge.get(tab, -PURGE_EVERY_SECONDS) < PURGE_EVERY_SECONDS:
        return
    _last_purge[tab] = now
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.USAGE_RETENTION_DAYS)
    expired = 0
    for stamp in worksheet.col_values(1)[1:]:
        try:
            if datetime.fromisoformat(stamp) >= cutoff:
                break
        except ValueError:
            break
        expired += 1
    if expired:
        worksheet.delete_rows(2, expired + 1)
        log.info("Deleted %d expired row(s) from the Google Sheet (%s).", expired, tab)
