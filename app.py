"""Web chat interface:  streamlit run app.py

Visitors choose a legal act on the left and ask about it. Every question is
also searched across all the acts in the library; when another act probably
covers it better, the reply says so and offers to switch to it.

The author's views (topics users asked to have added, usage statistics) are
in the terminal:
    python -m eu_law_nli.feedback
    python -m eu_law_nli.usage
"""
from __future__ import annotations

import hmac
import time

import streamlit as st

from eu_law_nli import config, live, sources
from eu_law_nli.engine import Engine, Reply, Session  # noqa: F401 (Reply: type hints)
from eu_law_nli.i18n import Localiser, document_strings, format_date
from eu_law_nli.library import Library
from eu_law_nli.llm import AnthropicLLM, LLMError
from eu_law_nli.registry import list_documents, load_document

st.set_page_config(page_title="EU law reference assistant", page_icon="⚖️")


def password_gate() -> None:
    """Ask for the shared test password when NLI_APP_PASSWORD is set (on
    Streamlit Community Cloud, in the app's Secrets). Stops the page until
    the right password is entered; nothing else loads or calls the model."""
    expected = config.APP_PASSWORD
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("EU law reference assistant")
    st.caption("This is a test version. Please enter the password you were given.")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("That password is not correct.")
    st.stop()


password_gate()


def text_key(doc_id: str) -> tuple[str, str, float]:
    """Which text of an act is current: its latest version and when it was built."""
    doc = load_document(doc_id)
    version = doc.version("latest").id
    return doc_id, version, live.text_stamp(doc_id, version, doc.reference_language)


@st.cache_resource(show_spinner="Indexing the library…", max_entries=2)
def get_library(keys: tuple) -> Library:
    # Keyed on every act's current text, so an adopted or rebuilt text is re-indexed.
    return Library([key[0] for key in keys])


@st.cache_resource(show_spinner="Loading the act…", max_entries=3)
def get_engine(key: tuple, library_keys: tuple) -> Engine:
    document, version, _ = key
    return Engine(AnthropicLLM(), document, version, library=get_library(library_keys))


def browser_language() -> str:
    return (st.context.locale or "en").split("-")[0].lower()


documents = list_documents()
all_docs = [load_document(d) for d in documents]
default_id = config.DEFAULT_DOCUMENT if config.DEFAULT_DOCUMENT in documents else documents[0]
previous: Session | None = st.session_state.get("session")
# The engine is not loaded yet, so these labels come from the saved translations.
ui = Localiser(None, extra=document_strings(all_docs)).strings(
    previous.language if previous else browser_language())

# Compare each act's stored text with the published one, in background
# threads: at start, then every LIVE_CHECK_HOURS. Nobody waits for it.
for d in documents:
    live.start_background(d)

if "doc_select" not in st.session_state:
    st.session_state["doc_select"] = default_id
with st.sidebar:
    # Shown even with one act, so more acts can be added to documents/ later.
    doc_id = st.selectbox(ui["legal_act_label"], documents, key="doc_select",
                          format_func=lambda d: ui.get(f"{d}.name", d))

try:
    if live.is_checking(doc_id) and st.session_state.get("engine_key", ("",))[0] == doc_id:
        engine_key = st.session_state["engine_key"]  # keep the current text while files change
    else:
        engine_key = text_key(doc_id)
    st.session_state["engine_key"] = engine_key
    library_keys = tuple(text_key(d) for d in documents)
    engine = get_engine(engine_key, library_keys)
except (LLMError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

if st.session_state.get("document") != doc_id or "session" not in st.session_state:
    st.session_state["document"] = doc_id
    session = Session(interface="web")
    carried = st.session_state.pop("carry_language", None)  # switching acts keeps the language
    if carried:
        session.language, session.language_name = carried
    else:
        # Until the user writes, show the page in the browser's language when the
        # act is loaded in it (its wording is then translated already).
        browser = browser_language()
        if browser.upper() in engine.corpora:
            session.language = browser
    st.session_state["session"] = session
    st.session_state["shown"] = []
session: Session = st.session_state["session"]
shown: list[tuple[str, object]] = st.session_state["shown"]
strings = engine.strings(session)


def show_reply(reply: Reply, latest: bool) -> None:
    """One assistant reply: summary and explanation first, the exact wording
    collapsed, then notes, a suggested act, the consent question and the
    small print."""
    if reply.summary:
        st.markdown(f"**{reply.summary}**")
    if reply.details:
        st.markdown(reply.details)
    if reply.wording:
        with st.expander(strings["show_wording"]):
            st.markdown(reply.wording)
    for note in reply.notes:
        st.caption(note)
    suggestion = getattr(reply, "suggestion", None)
    if suggestion and latest:
        name = strings.get(f"{suggestion['document']}.name", suggestion["name"])
        st.button(strings["switch_button"].format(document=name), key="switch",
                  on_click=_switch, args=(suggestion["document"], suggestion["question"]))
    if reply.consent_question:
        st.markdown(reply.consent_question)
        if latest and session.consent == "asked":
            yes, no, _ = st.columns([1, 1, 4])
            yes.button(strings["yes"], key="consent_yes", on_click=_choose, args=(True,))
            no.button(strings["no"], key="consent_no", on_click=_choose, args=(False,))
    if reply.followup:
        st.markdown(reply.followup)
    st.caption(f"{reply.disclaimer}  \n{reply.source}")


def _choose(agree: bool) -> None:
    st.session_state["consent_choice"] = agree


def _ask(text: str) -> None:
    st.session_state["queued"] = text


def _switch(document: str, question: str) -> None:
    """Change the selected act and ask the same question there."""
    st.session_state["doc_select"] = document
    st.session_state["queued"] = question
    st.session_state["carry_language"] = (session.language, session.language_name)


with st.sidebar:
    version = engine.version
    st.caption(engine.doc.citation)
    version_text = strings.get(f"version_line_{version.kind}",
                               strings["version_line_document"]).format(
        date=format_date(version.date))
    state = live.status(doc_id)
    if config.LIVE_CHECK and state.get("ok") and state.get("version") == version.id:
        version_text += ", " + strings["latest_note"].format(
            checked=format_date(state.get("last_ok_at") or state["checked_at"]))
    st.write(version_text)
    lang = session.language.upper() if session.language.upper() in engine.corpora \
        else engine.reference.lang
    official = sources.link(engine.doc, version, lang)
    if official:
        st.link_button(strings["open_eurlex"], official)
    if st.button(strings["new_conversation"]):
        for key in ("session", "shown", "queued", "consent_choice"):
            st.session_state.pop(key, None)
        st.rerun()

st.title(strings["page_title"])
st.caption(strings["page_intro"])
if config.USAGE_LOG:
    st.caption(strings["audit_notice"].format(days=config.USAGE_RETENTION_DAYS))

for index, (role, item) in enumerate(shown):
    with st.chat_message(role):
        # Checked by content, not by class: Streamlit reloads edited modules,
        # after which a stored reply is an instance of the previous Reply class.
        if not isinstance(item, str):
            show_reply(item, latest=index == len(shown) - 1)
        else:
            st.markdown(item)

examples = [strings[key] for key in (f"{doc_id}.example_1", f"{doc_id}.example_2")
            if key in strings]
if not shown and examples:
    st.write(strings["examples_heading"])
    for i, example in enumerate(examples):
        st.button(example, key=f"example_{i}", on_click=_ask, args=(example,))

choice = st.session_state.pop("consent_choice", None)
typed = st.chat_input(strings["input_placeholder"])
prompt = typed or st.session_state.pop("queued", None)

if choice is not None:
    shown.append(("user", strings["yes"] if choice else strings["no"]))
    shown.append(("assistant", engine.consent(session, choice)))
    st.rerun()

if prompt:
    shown.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.caption(strings["thinking"])
        last_draw = [0.0]

        def progress(markdown: str) -> None:
            # Redraw at most ten times a second; the text grows word by word.
            now = time.monotonic()
            if now - last_draw[0] >= 0.1:
                placeholder.markdown(markdown + " ▌")
                last_draw[0] = now

        reply = engine.respond(prompt, session, on_progress=progress)
    shown.append(("assistant", reply))
    st.rerun()  # redraw with the final layout, translated into the conversation's language
