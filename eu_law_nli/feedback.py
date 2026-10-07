"""Subjects users agreed to record for the author: national law or other EU
acts their question needed and the assistant could not review.

Stored one JSON object per line in data/feedback/out_of_scope.jsonl. Each
entry has the question and a list of topics, each with its source
("national_law", "other_eu_law" or "other"). The reason is "out_of_scope"
(the act does not deal with the subject), "not_covered" (the act was searched
but does not address it) or "partly_outside" (answered, but parts of the
question depend on other law).

    python -m eu_law_nli.feedback        # print what has been collected
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config

SOURCE_LABELS = {"national_law": "national law", "other_eu_law": "other EU act", "other": "other"}


def record(document: str, version: str, language: str, question: str,
           question_en: str = "", reason: str = "", topics: list[dict] | None = None) -> dict:
    entry = {
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": document,
        "version": version,
        "language": language,
        "question": question,
        "question_en": question_en,
        "reason": reason,
        "topics": [{"topic_en": t.get("topic_en", ""), "topic": t.get("topic", ""),
                    "source": t.get("source", "other")} for t in topics or []],
    }
    config.FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_all() -> list[dict]:
    if not config.FEEDBACK_FILE.exists():
        return []
    lines = config.FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def describe(topic: dict) -> str:
    """One topic as "General Data Protection Regulation (other EU act)"."""
    name = topic.get("topic_en") or topic.get("topic") or ""
    return f"{name} ({SOURCE_LABELS.get(topic.get('source'), 'other')})"


def main() -> int:
    entries = read_all()
    if not entries:
        print("No topics have been recorded yet.")
        return 0
    for e in entries:
        english = f"\n    (EN) {e['question_en']}" if e.get("question_en") and e["question_en"] != e["question"] else ""
        topics = "".join(f"\n    -> {describe(t)}" for t in e.get("topics") or [])
        print(f"{e['received_at']}  [{e['document']} {e['version']}, {e['language']}, "
              f"{e.get('reason', '')}]\n    {e['question']}{english}{topics}")
    print(f"\n{len(entries)} request(s) in {config.FEEDBACK_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
