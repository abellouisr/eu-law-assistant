"""Usage log: one line per turn in data/usage/usage.jsonl.

Each entry holds the time, a random conversation id, the interface (web or
cli), the user's language, the message as typed, what kind of reply it got,
the articles quoted, how many quotations passed or failed the word-for-word
check, subjects flagged as outside the act, response time and tokens used.
The reply itself is not stored.

Entries older than config.USAGE_RETENTION_DAYS (90 by default) are deleted
automatically: whenever a turn is logged or the log is read, and the oldest
entry is past the limit, the file is rewritten without the expired entries.

    python -m eu_law_nli.usage             # summary of the whole log
    python -m eu_law_nli.usage --days 7    # summary of the last 7 days
    python -m eu_law_nli.usage --list 20   # the 20 most recent questions
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone

from . import config

_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _time_of(entry: dict) -> datetime:
    return datetime.fromisoformat(entry["time"])


def record(entry: dict) -> None:
    """Append one turn to the log, then drop expired entries if any are due."""
    if not config.USAGE_LOG:
        return
    entry = {"time": _now().isoformat(timespec="seconds"), **entry}
    with _lock:
        config.USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with config.USAGE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _purge_if_due()


def purge(now: datetime | None = None) -> int:
    """Delete entries older than the retention period. Returns how many went."""
    with _lock:
        return _purge(now)


def _purge_if_due() -> None:
    """Rewrite the file only when its oldest (first) entry has expired."""
    path = config.USAGE_FILE
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        first = fh.readline().strip()
    if first and _time_of(json.loads(first)) < _cutoff():
        _purge()


def _cutoff(now: datetime | None = None) -> datetime:
    return (now or _now()) - timedelta(days=config.USAGE_RETENTION_DAYS)


def _purge(now: datetime | None = None) -> int:
    path = config.USAGE_FILE
    if not path.exists():
        return 0
    cutoff = _cutoff(now)
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [l for l in lines if _time_of(json.loads(l)) >= cutoff]
    if len(kept) == len(lines):
        return 0
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(l + "\n" for l in kept), encoding="utf-8")
    os.replace(tmp, path)
    return len(lines) - len(kept)


def read_all(days: int | None = None) -> list[dict]:
    """All entries still within retention, optionally only the last ``days``."""
    purge()
    if not config.USAGE_FILE.exists():
        return []
    lines = config.USAGE_FILE.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(l) for l in lines if l.strip()]
    if days is not None:
        since = _now() - timedelta(days=days)
        entries = [e for e in entries if _time_of(e) >= since]
    return entries


def cost(entry: dict) -> float | None:
    """Estimated API cost of one turn in USD at Anthropic's list prices, when the
    model's price is known. Bedrock ids ("eu.anthropic.claude-...") are matched
    on the model name; Bedrock and Foundry may bill differently."""
    model = entry.get("model", "").split("anthropic.")[-1]
    price = config.PRICES.get(model)
    if price is None:
        return None
    return (entry.get("input_tokens", 0) * price[0]
            + entry.get("output_tokens", 0) * price[1]) / 1_000_000


def summary(entries: list[dict]) -> dict:
    kinds = Counter(e.get("kind", "") for e in entries)
    costs = [c for c in (cost(e) for e in entries) if c is not None]
    seconds = [e["seconds"] for e in entries if e.get("seconds") is not None]
    return {
        "turns": len(entries),
        "conversations": len({e.get("session") for e in entries}),
        "first": entries[0]["time"] if entries else "",
        "last": entries[-1]["time"] if entries else "",
        "per_day": Counter(e["time"][:10] for e in entries),
        "kinds": kinds,
        "languages": Counter(e.get("language", "") for e in entries),
        "interfaces": Counter(e.get("interface", "") for e in entries),
        "articles": Counter(a for e in entries for a in e.get("articles", [])),
        "quotes_kept": sum(e.get("quotes_kept", 0) for e in entries),
        "quotes_dropped": sum(e.get("quotes_dropped", 0) for e in entries),
        "outside_topics": Counter(t for e in entries for t in e.get("outside_topics", [])),
        "avg_seconds": sum(seconds) / len(seconds) if seconds else 0.0,
        "cost_usd": sum(costs),
    }


def _top(counter: Counter, n: int = 8) -> str:
    return ", ".join(f"{k or 'unknown'} {v}" for k, v in counter.most_common(n)) or "-"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, help="only the last N days")
    ap.add_argument("--list", type=int, metavar="N", help="also print the N most recent questions")
    args = ap.parse_args(argv)

    entries = read_all(args.days)
    if not entries:
        print("No usage recorded" + (f" in the last {args.days} days." if args.days else "."))
        return 0
    s = summary(entries)
    answered = s["kinds"]["answer"] + s["kinds"]["situation"]
    declined = s["kinds"]["out_of_scope"]
    checked = s["quotes_kept"] + s["quotes_dropped"]
    print(f"Usage {s['first'][:10]} to {s['last'][:10]} "
          f"(kept for {config.USAGE_RETENTION_DAYS} days, in {config.USAGE_FILE})\n")
    print(f"  Messages          {s['turns']} in {s['conversations']} conversation(s)")
    print(f"  Answered          {answered}   out of scope {declined}   "
          f"errors {s['kinds']['error']}   other {s['turns'] - answered - declined - s['kinds']['error']}")
    print(f"  Languages         {_top(s['languages'])}")
    print(f"  Interface         {_top(s['interfaces'])}")
    print(f"  Articles quoted   {_top(s['articles'], 10)}")
    if checked:
        print(f"  Quotations        {s['quotes_kept']} verified, {s['quotes_dropped']} rejected "
              f"({100 * s['quotes_dropped'] / checked:.0f}% rejected)")
    print(f"  Outside the Code  {_top(s['outside_topics'], 5)}")
    print(f"  Response time     {s['avg_seconds']:.1f} s on average")
    print(f"  Estimated cost    ${s['cost_usd']:.2f}")
    print(f"  Per day           {', '.join(f'{d} {n}' for d, n in sorted(s['per_day'].items()))}")
    if args.list:
        print(f"\nMost recent questions:")
        for e in entries[-args.list:]:
            print(f"  {e['time'][:16].replace('T', ' ')}  [{e.get('language', '')}, "
                  f"{e.get('kind', '')}]  {e.get('question', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
