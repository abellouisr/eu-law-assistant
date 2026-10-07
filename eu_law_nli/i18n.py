"""Fixed wording (disclaimer, out-of-scope notice, prompts) in the user's language.

The English text below is the master copy. The first time a language is
used, the model translates the whole set once and the result is saved to
data/i18n/<code>.json. Those files are plain JSON: have a native speaker
review the disclaimer wording and edit it there.

Each file also keeps, under "_source", the English each entry was translated
from. When an English entry here changes, only that entry is translated again,
so reviewed wording for the other entries is kept.
"""
from __future__ import annotations

import json
import re

from . import config
from .llm import LLM

STRINGS: dict[str, str] = {
    "disclaimer": (
        "Disclaimer: I am a reference assistant, not a legal advisor. You are responsible "
        "for reviewing this response before relying on it or sharing it."
    ),
    "situation_invite": (
        "Would you like to share a situation you are dealing with? I can compare it with "
        "this act and indicate which rules may apply and how."
    ),
    "situation_prompt": (
        "Please describe the situation: who is involved, what has happened, and what you "
        "would like to know. Leave out names and other personal details."
    ),
    "situation_followup": (
        "If you can add detail on the points above, I can refine this comparison."
    ),
    "out_of_scope": (
        "Your question concerns the following subject: {topics}. As of {date}, the selected "
        "act ({document}) does not deal with it, and I cannot review national laws or acts "
        "that are not in this library. Please check the national law or the EU act that "
        "covers this subject before relying on any advice."
    ),
    "out_of_scope_plain": (
        "As of {date}, the selected act ({document}) does not deal with this question, and I "
        "cannot review national laws or acts that are not in this library. Please check the "
        "national law or the EU act that covers this subject before relying on any advice."
    ),
    "not_covered": (
        "The selected act ({document}) deals with this subject, but its text does not answer "
        "this specific question. The provisions that come closest are listed below; the "
        "point may be settled by national law, a Commission act or the contract."
    ),
    "closest_heading": "Closest provisions",
    "out_of_scope_library": (
        "The selected act ({document}) does not deal with this question, but another act "
        "in this library does."
    ),
    "limits_notice": (
        "This also depends on the following, which I cannot review: {topics}. Please check "
        "the national law or the other EU acts that apply before relying on this answer."
    ),
    "consent_question": (
        "Shall I record subjects like this for the author of this assistant, so they can "
        "be added in a future version? I will ask only once in this conversation."
    ),
    "recorded_note": "Recorded for the author: {topics}.",
    "topics_fallback": "a subject outside this act",
    "handoff_yes": (
        "Thank you. I have recorded this for the author: {topics}. For the rest of this "
        "conversation, I will record similar subjects without asking again."
    ),
    "handoff_no": "Understood. I will not record anything in this conversation.",
    "error_busy": (
        "The assistant is very busy right now. Please wait a minute and ask again."
    ),
    "error_setup": (
        "The assistant cannot reach its AI service because of a configuration problem. "
        "Please let the person who shared it with you know."
    ),
    "error_generic": (
        "Sorry, something went wrong while preparing the answer. Please try again, or "
        "rephrase your question."
    ),
    "page_title": "EU law reference assistant",
    "page_intro": (
        "Choose a legal act on the left and ask a question in any EU language. Answers "
        "quote the act and link to the official text. Reference assistant, not a legal "
        "advisor: review every response before relying on it or sharing it."
    ),
    "audit_notice": (
        "Audit log: during this trial, the questions are kept for {days} days for testing "
        "and improvement, then deleted automatically. Nothing that identifies you is "
        "recorded: no name, account, IP address or device details. Please do not include "
        "personal or confidential details in your questions."
    ),
    "input_placeholder": "Ask about the selected act, or describe a situation",
    "examples_heading": "Try asking:",
    "show_wording": "Show the exact wording",
    "thinking": "Reading the text…",
    "suggest_act": (
        "The {document} probably covers this in more detail (for example {provisions})."
    ),
    "switch_button": "Switch to {document} and ask there",
    "yes": "Yes",
    "no": "No",
    "new_conversation": "New conversation",
    "legal_act_label": "Legal act",
    "version_line_consolidated": "Consolidated version of {date}",
    "version_line_original": "Official Journal version of {date}",
    "version_line_document": "Version of {date}",
    "latest_note": "the latest published version (checked {checked})",
    "open_eurlex": "Open the official text",
    "anything_else": "Feel free to ask another question about this act: {document}.",
    "greeting": (
        "Hello. I answer questions about one act: {document}. I quote its exact wording and "
        "point you to the provisions concerned."
    ),
    "quotes_heading": "Exact wording",
    "references_heading": "References",
    "see_also_heading": "Provisions consulted",
    "quote_language_note": (
        "The quotations are from the official {language} text, because this act is not "
        "loaded in your language."
    ),
    "no_verified_quote": (
        "I could not confirm an exact quotation for this answer, so none is shown. Please "
        "read the provisions listed below directly."
    ),
    "source_note": "Source: {citation}, {version_kind} text of {date}",
    "live_checked": "checked against the published text {checked}",
    "live_failed": "not checked against the published text since {checked}; a newer "
                   "version may exist",
    "live_newer": "a newer version of {date} has been published and will be used next",
    "version_original": "Official Journal",
    "version_consolidated": "consolidated",
    "version_document": "published",
    "recital": "Recital",
}

def format_date(iso: str) -> str:
    """2024-10-18 (or a full timestamp) -> 18.10.2024, the style EUR-Lex uses."""
    return f"{iso[8:10]}.{iso[5:7]}.{iso[:4]}" if len(iso) >= 10 else iso


_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")

TRANSLATE_SYSTEM = (
    "You translate the fixed interface texts of a legal reference assistant. Translate "
    "each value into the target language, formally and faithfully, without adding or "
    "softening anything. Keep every placeholder in curly braces exactly as written. "
    "Return every key."
)


def _schema(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": keys,
    }


def document_strings(documents) -> dict[str, str]:
    """Per-document wording to translate along with the fixed texts: each
    document's name and its example questions ("eecc.name", "eecc.example_1")."""
    out: dict[str, str] = {}
    for doc in documents:
        out[f"{doc.id}.name"] = doc.short_name
        for i, example in enumerate(doc.examples[:2], start=1):
            out[f"{doc.id}.example_{i}"] = example
    return out


class Localiser:
    def __init__(self, llm: LLM | None, extra: dict[str, str] | None = None) -> None:
        self.llm = llm
        self.master = {**STRINGS, **(extra or {})}  # the English to translate from
        self._cache: dict[str, dict[str, str]] = {"en": dict(self.master)}

    def strings(self, code: str, language_name: str = "") -> dict[str, str]:
        code = (code or "en").lower()[:5]
        if code in self._cache:
            return self._cache[code]
        path = config.I18N_DIR / f"{code}.json"
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        source = saved.get("_source", {})
        stale = [k for k in self.master if k not in saved or source.get(k) != self.master[k]]
        if not stale:
            self._cache[code] = self._merge(saved)
            return self._cache[code]
        translated = self._translate(code, language_name, stale)
        if translated is None:
            # Keep what is saved for now (not cached, so the next turn tries again).
            return self._merge({k: v for k, v in saved.items() if k not in stale})
        fresh = self._merge({**saved, **translated})
        # Entries this instance does not know (other documents) are kept as they are.
        record = {**saved, **fresh, "_source": {**source, **self.master}}
        config.write_text_atomic(path, json.dumps(record, ensure_ascii=False, indent=2))
        self._cache[code] = self._merge(record)
        return self._cache[code]

    def _translate(self, code: str, language_name: str, keys: list[str]) -> dict | None:
        if self.llm is None:
            return None
        try:
            return self.llm.json(
                TRANSLATE_SYSTEM,
                f"Target language: {language_name or code} ({code})\n\n"
                + json.dumps({k: self.master[k] for k in keys}, ensure_ascii=False, indent=2),
                _schema(keys), "translate_interface",
            )
        except Exception:
            return None

    def _merge(self, candidate: dict) -> dict[str, str]:
        """Use a translation only if it kept the placeholders; else keep English."""
        out = dict(self.master)
        for key, english in self.master.items():
            value = candidate.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            if sorted(_PLACEHOLDER.findall(value)) == sorted(_PLACEHOLDER.findall(english)):
                out[key] = value.strip()
        return out
