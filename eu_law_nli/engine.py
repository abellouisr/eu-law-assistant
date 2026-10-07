"""The conversation engine.

One turn:
  1. analyse   - language, intent, in scope?, search terms        (model call)
  2. retrieve  - find the passages of the act                     (local)
  3. answer    - write a plain summary, the explanation and pick
                 quotations; streamed, so the web app can show it as
                 it is written                                    (model call)
  4. verify    - keep only quotations found word for word in the act
  5. finish    - add references, notes, the disclaimer and the source

Every reply leaves through ``_finish``, which adds the disclaimer to the
first reply of a conversation (users also accept it when they log in).

Subjects outside the act (national law, other EU acts) are recorded for the
author in the topics log (see feedback.py), without asking the user.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from . import config, feedback, live, prompts, sources, usage
from .corpus import Chunk, Corpus, reference
from .i18n import Localiser, document_strings, format_date
from .llm import LLM, LLMBusy, LLMError, LLMSetupError
from .parser import Provision, normalise
from .registry import Document, load_document
from .retriever import BM25Retriever

if TYPE_CHECKING:
    from .library import Library

_LOOSE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‑": "-",
})
MIN_QUOTE_CHARS = 15
log = logging.getLogger("eu_law_nli")


@dataclass
class Passage:
    id: str
    provision: Provision
    text: str
    lang: str


@dataclass
class Quote:
    text: str  # exact wording as stored from EUR-Lex
    reference: str  # e.g. "Article 61(2)"
    url: str
    provision_id: str


@dataclass
class Reply:
    text: str  # complete markdown: body, then disclaimer and source
    kind: str  # answer | situation | out_of_scope | decline | greeting | prompt | error
    language: str = "en"
    body: str = ""  # the reply without disclaimer and source line
    quotes: list[Quote] = field(default_factory=list)
    dropped_quotes: list[str] = field(default_factory=list)
    # The same reply in parts, for interfaces that lay it out themselves:
    summary: str = ""  # one or two plain sentences (answers only)
    details: str = ""  # the explanation, or the whole message for other replies
    wording: str = ""  # exact quotations and references (shown collapsed on the web)
    notes: list[str] = field(default_factory=list)
    followup: str = ""
    disclaimer: str = ""  # only on the first reply of a conversation
    source: str = ""
    outside_topics: list[dict] = field(default_factory=list)
    # Another act that probably covers the question better: {"document": id,
    # "name": ..., "provisions": [...], "question": the user's message}
    suggestion: dict | None = None


@dataclass
class Session:
    history: list[tuple[str, str]] = field(default_factory=list)
    language: str = "en"
    language_name: str = "English"
    document_name: str = ""  # the act's name in the user's language
    disclaimer_shown: bool = False  # the disclaimer comes with the first reply only
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])  # random, for the usage log
    interface: str = ""  # "web" or "cli", for the usage log


def continue_in(previous: Session, document_name: str, interface: str = "") -> Session:
    """A new session for another act that keeps the conversation going: the
    same language, the recent exchanges (so follow-up questions keep their
    meaning) and a note telling the model that the act has changed."""
    session = Session(interface=interface or previous.interface)
    session.disclaimer_shown = previous.disclaimer_shown
    session.language, session.language_name = previous.language, previous.language_name
    session.document_name = ""  # set again from the new act's analysis
    session.history = list(previous.history[-2 * config.HISTORY_TURNS:])
    session.history.append((
        "note", f"The conversation now continues in another act: {document_name}. Earlier "
                "answers came from the act named in them; answer from this act only."))
    return session


def locate(haystack: str, needle: str) -> str | None:
    """Find ``needle`` in ``haystack`` ignoring whitespace and quote-mark style.

    Returns the matching text exactly as it appears in ``haystack``, so what
    is displayed is always the stored wording and never the model's retyping.
    """
    h = normalise(haystack)
    n = normalise(needle).strip('"').strip()
    if len(n) < MIN_QUOTE_CHARS:
        return None
    idx = h.translate(_LOOSE).find(n.translate(_LOOSE))
    return h[idx: idx + len(n)] if idx >= 0 else None


def partial_field(raw: str, key: str) -> str:
    """The value of a string field in JSON that is still arriving: whatever
    has been received so far, unescaped. Empty until the field has started."""
    marker = raw.find(f'"{key}"')
    if marker < 0:
        return ""
    start = raw.find('"', raw.find(":", marker) + 1)
    if start < 0:
        return ""
    out, i = [], start + 1
    while i < len(raw):
        ch = raw[i]
        if ch == '"':
            break
        if ch == "\\":
            if i + 1 >= len(raw):
                break
            nxt = raw[i + 1]
            if nxt == "u":
                if i + 6 > len(raw):
                    break
                try:
                    out.append(chr(int(raw[i + 2:i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append({"n": "\n", "t": "\t", "r": "", "b": "", "f": ""}.get(nxt, nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _utc_minutes(stamp: str) -> str:
    """2026-10-07T08:34:43+00:00 -> 07.10.2026 08:34 UTC."""
    return f"{format_date(stamp)} {stamp[11:16]} UTC" if len(stamp) >= 16 else ""


@lru_cache(maxsize=8)
def _corpus(document: str, version: str, lang: str) -> Corpus:
    """Another act's text in one language, for labels in suggestions."""
    return Corpus.load(document, version, lang)


def _clean_topics(raw) -> list[dict]:
    """Subjects outside the act, as the model listed them, with empty ones removed."""
    topics = []
    for t in raw or []:
        if isinstance(t, dict) and str(t.get("topic") or t.get("topic_en") or "").strip():
            topics.append({"topic": str(t.get("topic") or t.get("topic_en")).strip(),
                           "topic_en": str(t.get("topic_en") or t.get("topic")).strip(),
                           "source": t.get("source") or "other"})
    return topics


class Engine:
    def __init__(self, llm: LLM, document: str | Document | None = None,
                 version: str | None = None, library: "Library | None" = None) -> None:
        self.llm = llm
        self.doc = document if isinstance(document, Document) else load_document(
            document or config.DEFAULT_DOCUMENT)
        self.version = self.doc.version(version)
        langs = Corpus.available_languages(self.doc.id, self.version.id)
        ref_lang = self.doc.reference_language.upper()
        if ref_lang not in langs:
            raise FileNotFoundError(
                f"The {ref_lang} text of '{self.doc.id}' (version {self.version.id}) has not "
                f"been ingested yet. Run: python -m eu_law_nli.ingest {self.doc.id}"
            )
        self.corpora = {l: Corpus.load(self.doc.id, self.version.id, l) for l in langs}
        self.reference = self.corpora[ref_lang]
        self.retriever = BM25Retriever(self.reference)
        # The library searches every act, to suggest a better-matching one.
        self.library = library
        docs = {self.doc.id: self.doc, **(library.documents if library else {})}
        self.localiser = Localiser(llm, extra=document_strings(docs.values()))
        self._outline = self.reference.outline()

    def strings(self, session: Session) -> dict[str, str]:
        """The fixed wording in the session's language (also used by the web page)."""
        return self.localiser.strings(session.language, session.language_name)

    # ------------------------------------------------------------------
    def respond(self, message: str, session: Session,
                on_progress: Callable[[str], None] | None = None) -> Reply:
        """Answer one message. ``on_progress`` receives the summary and the
        explanation as markdown while they are being written."""
        message = message.strip()
        started, tokens_before = time.monotonic(), self._tokens()
        error = ""
        try:
            reply = self._respond(message, session, on_progress)
        except LLMError as exc:
            # The technical detail goes to the usage log and the server log
            # (on Streamlit Community Cloud: Manage app), not to the user.
            error = str(exc)
            log.error("Model call failed: %s", error)
            strings = self.strings(session)
            key = ("error_busy" if isinstance(exc, LLMBusy)
                   else "error_setup" if isinstance(exc, LLMSetupError) else "error_generic")
            reply = self._finish(strings, session, "error", details=strings[key])
        session.history.append(("user", message))
        session.history.append(("assistant", reply.body))
        self._log(message, session, reply, time.monotonic() - started, tokens_before, error)
        return reply

    def _tokens(self) -> tuple[int, int]:
        counter = getattr(self.llm, "tokens", None)
        return counter() if callable(counter) else (0, 0)

    def _log(self, message: str, session: Session, reply: Reply, seconds: float,
             tokens_before: tuple[int, int], error: str) -> None:
        """One line in the usage log. A logging failure never affects the reply."""
        tokens_after = self._tokens()
        try:
            usage.record({
                "session": session.id,
                "interface": session.interface,
                "document": self.doc.id,
                "version": self.version.id,
                "language": session.language,
                "kind": reply.kind,
                "question": message,
                "articles": sorted({q.provision_id for q in reply.quotes}),
                "quotes_kept": len(reply.quotes),
                "quotes_dropped": len(reply.dropped_quotes),
                "outside_topics": [t.get("topic_en", "") for t in reply.outside_topics],
                "seconds": round(seconds, 1),
                "model": getattr(self.llm, "model", ""),
                "provider": getattr(self.llm, "provider", ""),
                "input_tokens": tokens_after[0] - tokens_before[0],
                "output_tokens": tokens_after[1] - tokens_before[1],
                **({"error": error} if error else {}),
            })
        except Exception:
            pass

    def _respond(self, message: str, session: Session,
                 on_progress: Callable[[str], None] | None) -> Reply:
        analysis = self._analyse(message, session)
        session.language = (analysis.get("language_code") or session.language or "en").lower()
        session.language_name = analysis.get("language_name") or session.language_name
        strings = self.strings(session)
        intent = analysis.get("intent", "question")
        session.document_name = (str(analysis.get("document_name") or "").strip()
                                 or session.document_name or self.doc.short_name)
        doc_name = session.document_name

        # A yes or no to the invitation to describe a situation.
        if intent == "consent_yes":
            return self._finish(strings, session, "prompt", details=strings["situation_prompt"])
        if intent == "consent_no":
            return self._finish(strings, session, "decline",
                                details=strings["anything_else"].format(document=doc_name))
        if intent == "greeting":
            return self._finish(strings, session, "greeting",
                                details=strings["greeting"].format(document=doc_name),
                                followup=strings["situation_invite"])

        question_en = analysis.get("standalone_question") or message
        if not analysis.get("in_scope", True):
            return self._out_of_scope(message, question_en, "out_of_scope",
                                      analysis.get("outside_topics") or [], session, strings,
                                      self._suggest(analysis, message, question_en, False,
                                                    session))

        passages = self._retrieve(analysis, question_en, session.language)
        mode = "situation" if intent == "situation" else "question"
        answer = self._answer(message, question_en, passages, mode, session, on_progress)
        topics = _clean_topics(answer.get("outside_topics"))
        if answer.get("status") != "answered" or not (answer.get("answer") or "").strip():
            return self._out_of_scope(message, question_en, "not_covered", topics,
                                      session, strings,
                                      self._suggest(analysis, message, question_en, False,
                                                    session), passages)

        quotes, dropped = self._verify(answer.get("quotes") or [], passages, strings)
        wording = self._wording(quotes, passages, strings)
        notes = []
        if session.language.upper() not in {p.lang for p in passages}:
            notes.append(strings["quote_language_note"].format(language=self.reference.lang))
        if topics:
            notes.append(strings["limits_notice"].format(
                document=doc_name, topics=self._topic_list(topics, strings)))
        self._record(message, question_en, "partly_outside", topics, session)
        suggestion = self._suggest(analysis, message, question_en, True, session)
        if suggestion:
            notes.append(self._suggestion_note(suggestion, strings))
        followup = (strings["situation_followup"] if mode == "situation"
                    else strings["situation_invite"])
        reply = self._finish(
            strings, session, mode if mode == "situation" else "answer",
            summary=(answer.get("summary") or "").strip(), details=answer["answer"].strip(),
            wording=wording, notes=notes, followup=followup, topics=topics,
            suggestion=suggestion)
        reply.quotes, reply.dropped_quotes = quotes, dropped
        return reply

    # ------------------------------------------------------------------
    def _record(self, message: str, question_en: str, reason: str,
                topics: list[dict], session: Session) -> None:
        """Note subjects outside the act in the topics log for the author."""
        if not topics:
            return
        try:
            feedback.record(self.doc.id, self.version.id, session.language, message, question_en,
                            reason, topics)
        except OSError as exc:  # a logging failure never affects the reply
            log.warning("Could not record topics: %s", exc)

    def _conversation(self, session: Session) -> str:
        turns = session.history[-2 * config.HISTORY_TURNS:]
        if not turns:
            return "(no earlier messages)"
        return "\n".join(f"{role.upper()}: {text[:1500]}" for role, text in turns)

    def _analyse(self, message: str, session: Session) -> dict:
        pending = "an invitation to share a situation, if one was made"
        system = prompts.ANALYSE_SYSTEM.format(
            title=self.doc.title, scope=self.doc.scope, outline=self._outline,
            previous_language=f"{session.language_name} ({session.language})", pending=pending,
            outside_topics_rule=prompts.OUTSIDE_TOPICS_RULE.format(
                topic_language="the language of the latest message"),
            library_acts=self._library_acts(),
            short_name=self.doc.short_name,
        )
        user = (f"<conversation>\n{self._conversation(session)}\n</conversation>\n\n"
                f"<latest_message>\n{message}\n</latest_message>")
        return self.llm.json(system, user, prompts.ANALYSE_SCHEMA, "classify_message")

    def _library_acts(self) -> str:
        others = [d for d in (self.library.documents.values() if self.library else [])
                  if d.id != self.doc.id]
        return "\n".join(f"{d.id}: {d.short_name}. {d.scope}" for d in others) or "(none)"

    def _retrieve(self, analysis: dict, question_en: str, language: str) -> list[Passage]:
        queries = [q for q in (analysis.get("search_queries") or []) if q.strip()]
        queries.append(question_en)
        chunks: list[Chunk] = []
        for pid in analysis.get("referenced_provisions") or []:
            chunks.extend(self.retriever.chunks_of(str(pid).strip()))
        chunks.extend(hit.chunk for hit in self.retriever.search_many(queries, k=config.TOP_K))

        target = self.corpora.get(language.upper(), self.reference)
        passages: list[Passage] = []
        seen: set[tuple[str, str]] = set()
        used = 0
        for chunk in chunks:
            provision, text, lang = self._in_language(chunk, target)
            key = (provision.id, text[:80])
            if key in seen:
                continue
            if used + len(text) > config.MAX_CONTEXT_CHARS and passages:
                break
            seen.add(key)
            used += len(text)
            passages.append(Passage(f"P{len(passages) + 1}", provision, text, lang))
        return passages

    def _in_language(self, chunk: Chunk, target: Corpus) -> tuple[Provision, str, str]:
        """The same passage in the user's language when that text is loaded."""
        ref_p = self.reference.get(chunk.provision_id)
        if target is self.reference:
            return ref_p, chunk.text, self.reference.lang
        tgt_p = target.get(chunk.provision_id)
        if tgt_p is None or not tgt_p.blocks:
            return ref_p, chunk.text, self.reference.lang
        if len(tgt_p.blocks) == len(ref_p.blocks):
            blocks = tgt_p.blocks[chunk.start:chunk.end]
        else:  # layouts differ between languages: match on paragraph numbers
            labels = {(b.para, b.point) for b in ref_p.blocks[chunk.start:chunk.end]} - {("", "")}
            blocks = [b for b in tgt_p.blocks if (b.para, b.point) in labels] or tgt_p.blocks
        text = "\n".join(b.text for b in blocks)[:6000]
        return tgt_p, text, target.lang

    def _answer(self, message: str, question_en: str, passages: list[Passage],
                mode: str, session: Session,
                on_progress: Callable[[str], None] | None = None) -> dict:
        if not passages:
            return {"status": "not_covered", "summary": "", "answer": "", "quotes": []}
        rules = prompts.SITUATION_RULES if mode == "situation" else prompts.QUESTION_RULES
        system = prompts.ANSWER_SYSTEM.format(
            short_name=self.doc.short_name, outside_topics_rule=prompts.OUTSIDE_TOPICS_RULE.format(
                topic_language=session.language_name),
            title=self.doc.title, citation=self.doc.citation,
            version_kind=self.version.kind, version_date=self.version.date,
            language_name=session.language_name,
            mode_rules=rules.format(language_name=session.language_name),
        )
        blocks = []
        for p in passages:
            heading = " > ".join(p.provision.path)
            blocks.append(
                f'<passage id="{p.id}" reference="{reference(p.provision)}" '
                f'title="{p.provision.title}" part="{heading}" language="{p.lang}">\n'
                f"{p.text}\n</passage>"
            )
        user = (
            f"<conversation>\n{self._conversation(session)}\n</conversation>\n\n"
            f"<passages>\n" + "\n".join(blocks) + "\n</passages>\n\n"
            f"<request>\n{message}\n</request>\n"
            f"<request_in_english>\n{question_en}\n</request_in_english>"
        )
        stream = None
        if on_progress is not None:
            def stream(raw: str) -> None:
                if '"not_covered"' in raw[:60]:
                    return
                summary, details = partial_field(raw, "summary"), partial_field(raw, "answer")
                if summary or details:
                    on_progress(f"**{summary}**\n\n{details}" if details else f"**{summary}**")
        return self.llm.json(system, user, prompts.ANSWER_SCHEMA, "write_answer", on_text=stream)

    def _verify(self, raw_quotes: list[dict], passages: list[Passage],
                strings: dict) -> tuple[list[Quote], list[str]]:
        by_id = {p.id: p for p in passages}
        quotes: list[Quote] = []
        dropped: list[str] = []
        for raw in raw_quotes:
            text = str(raw.get("text", ""))
            named = by_id.get(str(raw.get("passage_id", "")))
            candidates = ([named] if named else []) + [p for p in passages if p is not named]
            found = None
            for passage in candidates:
                # The whole provision is searched, so a quotation that runs a
                # little past the retrieved passage is still accepted.
                exact = locate(passage.provision.text, text)
                if exact:
                    found = (passage, exact)
                    break
            if not found:
                dropped.append(text)
                continue
            passage, exact = found
            if any(q.text == exact for q in quotes):
                continue
            quotes.append(Quote(
                text=exact,
                reference=self._reference(passage.provision, exact, strings),
                url=sources.link(self.doc, self.version, passage.lang, passage.provision),
                provision_id=passage.provision.id,
            ))
        return quotes, dropped

    @staticmethod
    def _reference(provision: Provision, quote: str, strings: dict) -> str:
        if provision.kind == "recital":
            return f"{strings['recital']} {provision.number}"
        return reference(provision, quote)

    def _wording(self, quotes: list[Quote], passages: list[Passage], strings: dict) -> str:
        """Exact quotations and the provisions they come from, as markdown."""
        parts = []
        if quotes:
            parts.extend(f"> “{q.text}”\n>\n> — " + (f"[{q.reference}]({q.url})" if q.url
                                                      else q.reference) for q in quotes)
            refs: dict[str, str] = {}
            for q in quotes:
                refs.setdefault(q.provision_id, q.url)
            heading = strings["references_heading"]
        else:
            parts.append(strings["no_verified_quote"])
            refs = {p.provision.id: sources.link(self.doc, self.version, p.lang, p.provision)
                    for p in passages[:5]}
            heading = strings["see_also_heading"]
        by_id = {p.provision.id: p.provision for p in passages}
        items = []
        for pid, url in refs.items():
            prov = by_id[pid]
            name = self._reference(prov, "", strings)
            label = f"{name}{' — ' + prov.title if prov.title else ''}"
            items.append(f"- [{label}]({url})" if url else f"- {label}")
        parts.append(f"**{heading}**\n" + "\n".join(items))
        return "\n\n".join(parts)

    def _suggest(self, analysis: dict, message: str, question_en: str,
                 answered: bool, session: Session) -> dict | None:
        """Another act in the library that matches the question clearly better.
        After an answer, the other act must score SUGGEST_RATIO times higher on
        keywords. When this act does not cover the question, the intake step
        must also have named that act as covering the subject, so that shared
        words alone ("contract" in a tax question) never trigger a suggestion."""
        if self.library is None or len(self.library) < 2:
            return None
        queries = [q for q in (analysis.get("search_queries") or []) if q.strip()]
        ranked = self.library.rank(queries + [question_en])
        current = next((m.score for m in ranked if m.document == self.doc.id), 0.0)
        if answered:
            other = next((m for m in ranked if m.document != self.doc.id), None)
            if other is None or other.score < max(config.SUGGEST_MIN_SCORE,
                                                   current * config.SUGGEST_RATIO):
                return None
        else:
            named = str(analysis.get("library_act") or "").strip()
            other = next((m for m in ranked if m.document == named), None)
            if not named or named == self.doc.id or other is None \
                    or other.score < config.SUGGEST_MIN_SCORE:
                return None
        return {"document": other.document, "name": other.short_name,
                "provisions": self._labels_in(other, session.language), "question": message}

    def _labels_in(self, match, language: str) -> list[str]:
        """The suggested provisions' labels in the user's language, when the
        other act is loaded in it ("Artikkel 13" rather than "Article 13")."""
        doc = self.library.documents[match.document]
        try:
            corpus = _corpus(doc.id, doc.version("latest").id, language.upper())
        except FileNotFoundError:
            return match.provisions
        recital = self.localiser.strings(language).get("recital", "Recital")
        labels = []
        for pid, english in zip(match.provision_ids, match.provisions):
            p = corpus.get(pid)
            if p is None:
                labels.append(english)
            elif p.kind == "recital":
                labels.append(f"{recital} {p.number}")
            else:
                labels.append(p.label)
        return labels

    @staticmethod
    def _suggestion_note(suggestion: dict, strings: dict) -> str:
        name = strings.get(f"{suggestion['document']}.name", suggestion["name"])
        return strings["suggest_act"].format(document=name,
                                             provisions=", ".join(suggestion["provisions"]))

    def _out_of_scope(self, message: str, question_en: str, reason: str, topics: list[dict],
                      session: Session, strings: dict, suggestion: dict | None = None,
                      passages: list[Passage] | None = None) -> Reply:
        topics = _clean_topics(topics)
        document = session.document_name or self.doc.short_name
        if suggestion:  # another act in the library covers it: point there, record nothing
            details = strings["out_of_scope_library"].format(document=document)
            return self._finish(strings, session, "out_of_scope", details=details,
                                notes=[self._suggestion_note(suggestion, strings)],
                                topics=topics, suggestion=suggestion)
        wording = ""
        if reason == "not_covered" and passages:
            # The act deals with the subject but not with this exact question:
            # say so, and point to the provisions that came closest.
            details = strings["not_covered"].format(document=document)
            wording = self._references(passages[:5], strings, strings["closest_heading"])
        elif topics:
            details = strings["out_of_scope"].format(
                date=format_date(date.today().isoformat()), document=document,
                topics=self._topic_list(topics, strings))
        else:
            details = strings["out_of_scope_plain"].format(
                date=format_date(date.today().isoformat()), document=document)
        # Nothing named: record the question itself as the subject.
        self._record(message, question_en, reason, topics or [
            {"topic": message[:200], "topic_en": question_en[:200], "source": "other"}], session)
        return self._finish(strings, session, "out_of_scope", details=details,
                            topics=topics, wording=wording)

    def _references(self, passages: list[Passage], strings: dict, heading: str) -> str:
        """A linked list of the provisions behind some passages, without quotations."""
        items: dict[str, str] = {}
        for p in passages:
            if p.provision.id in items:
                continue
            url = sources.link(self.doc, self.version, p.lang, p.provision)
            name = self._reference(p.provision, "", strings)
            label = f"{name}{' — ' + p.provision.title if p.provision.title else ''}"
            items[p.provision.id] = f"- [{label}]({url})" if url else f"- {label}"
        return f"**{heading}**\n" + "\n".join(items.values()) if items else ""

    @staticmethod
    def _topic_list(topics: list[dict], strings: dict) -> str:
        names = [t["topic"] for t in topics if t.get("topic")]
        return ", ".join(names) if names else strings["topics_fallback"]

    def _finish(self, strings: dict, session: Session, kind: str, *, summary: str = "",
                details: str = "", wording: str = "", notes: list[str] | None = None,
                followup: str = "", topics: list[dict] | None = None,
                suggestion: dict | None = None) -> Reply:
        """Single exit point: every reply gets the source line, and the first
        one of a conversation (an error message aside) the disclaimer."""
        source = strings["source_note"].format(
            citation=self.doc.citation,
            version_kind=strings.get(f"version_{self.version.kind}", self.version.kind),
            date=format_date(self.version.date),
        )
        checked = self._live_note(strings)
        source = f"{source} · {checked}." if checked else f"{source}."
        notes = notes or []
        disclaimer = ""
        if not session.disclaimer_shown and kind != "error":
            disclaimer, session.disclaimer_shown = strings["disclaimer"], True
        body = "\n\n".join(x for x in (
            f"**{summary}**" if summary else "", details, wording,
            *[f"_{n}_" for n in notes], followup) if x)
        small_print = f"_{disclaimer}_\n\n_{source}_" if disclaimer else f"_{source}_"
        text = f"{body}\n\n---\n{small_print}"
        return Reply(text=text, kind=kind, language=session.language, body=body,
                     summary=summary, details=details, wording=wording, notes=notes,
                     followup=followup, disclaimer=disclaimer, source=source,
                     outside_topics=topics or [], suggestion=suggestion)

    def _live_note(self, strings: dict) -> str:
        """When the text was last compared with the live EU version, if ever."""
        if not config.LIVE_CHECK:
            return ""
        state = live.status(self.doc.id)
        if not state:
            return ""
        last_ok = _utc_minutes(state.get("last_ok_at") or (
            state.get("checked_at", "") if state.get("ok") else ""))
        if state.get("ok") and state.get("version") == self.version.id:
            return strings["live_checked"].format(checked=last_ok)
        if state.get("ok"):  # the background check adopted a newer text meanwhile
            return strings["live_newer"].format(date=format_date(state.get("version", "")))
        return strings["live_failed"].format(checked=last_ok or "-")
