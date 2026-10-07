"""Chat with the assistant in a terminal:  python cli.py [document]

    /acts            list the acts in the library
    /switch <id>     change act (after a suggestion: /switch alone, asks there too)
    exit             leave
"""
from __future__ import annotations

import sys

from eu_law_nli import config, live
from eu_law_nli.engine import Engine, Session, continue_in
from eu_law_nli.library import Library
from eu_law_nli.llm import AnthropicLLM, LLMError
from eu_law_nli.registry import list_documents, load_document


def main() -> int:
    document = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_DOCUMENT
    # Checks each act's published text in the background; a text it adopts or
    # rebuilds is picked up before the next question.
    for doc_id in list_documents():
        live.start_background(doc_id)
    try:
        llm = AnthropicLLM()
        library = Library()
        engine = Engine(llm, document, library=library)
    except (LLMError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1
    stamp = _text_stamp(document)
    _banner(engine)
    if config.USAGE_LOG:
        print(f"Audit log: during this trial, the questions are kept for "
              f"{config.USAGE_RETENTION_DAYS} days for testing and improvement, then deleted\n"
              "automatically. Nothing that identifies you is recorded: no name, account, IP "
              "address or device details.\nPlease do not include personal or confidential "
              "details in your questions.")
    print()
    session = Session(interface="cli")
    suggestion = None
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            return 0
        if message == "/acts":
            for doc_id, doc in library.documents.items():
                mark = "*" if doc_id == engine.doc.id else " "
                print(f" {mark} {doc_id:10} {doc.short_name} ({doc.citation})")
            continue
        if message.startswith("/switch"):
            target = message[len("/switch"):].strip() or (suggestion or {}).get("document")
            if target not in library.documents:
                print(f"Unknown act '{target}'. Type /acts to see the list.")
                continue
            question = (suggestion or {}).get("question") if not message[7:].strip() else None
            document, suggestion = target, None
            engine, stamp = Engine(llm, document, library=library), _text_stamp(document)
            session = continue_in(session, engine.doc.short_name)
            _banner(engine)
            if not question:
                continue
            print(f"You: {question}")
            message = question
        if not live.is_checking(document) and _text_stamp(document) != stamp:
            library = Library()
            engine, stamp = Engine(llm, document, library=library), _text_stamp(document)
            print(f"(Now using the {engine.version.kind} text of {engine.version.date}.)")
        reply = engine.respond(message, session)
        print(f"\nAssistant:\n{reply.text}\n")
        suggestion = reply.suggestion
        if suggestion:
            print(f"(Type /switch to ask this in {suggestion['name']}.)\n")


def _banner(engine: Engine) -> None:
    langs = ", ".join(sorted(engine.corpora))
    print("EU LexRef — EU law reference assistant")
    print(f"{engine.doc.short_name} — {engine.version.kind} text of {engine.version.date} "
          f"(languages loaded: {langs})\nAsk in any language. /acts lists the library, "
          "/switch <id> changes act, 'exit' leaves.")


def _text_stamp(document: str) -> tuple[str, float]:
    """Which text is current: the latest version and when it was last built."""
    doc = load_document(document)
    version = doc.version("latest").id
    return version, live.text_stamp(document, version, doc.reference_language)


if __name__ == "__main__":
    raise SystemExit(main())
