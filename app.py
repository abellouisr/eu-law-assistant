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
import logging
import os
import time

import streamlit as st


def secrets_to_environment() -> None:
    """On Streamlit Community Cloud, settings such as ANTHROPIC_API_KEY and
    NLI_APP_PASSWORD are entered as the app's Secrets. Copy the top-level ones
    into the environment before the settings module reads it. Locally there
    is usually no secrets file, and .env is used instead."""
    try:
        if not st.secrets.load_if_toml_exists():
            return
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:  # a malformed secrets file must not stop the app here
        pass


secrets_to_environment()
# Errors from the model call appear in the server log (Community Cloud: Manage app).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from eu_law_nli import config, i18n, live, sources  # noqa: E402 (after the secrets are loaded)
from eu_law_nli.engine import Engine, Reply, Session, continue_in  # noqa: E402,F401
from eu_law_nli.i18n import Localiser, document_strings, format_date  # noqa: E402
from eu_law_nli.library import Library  # noqa: E402
from eu_law_nli.llm import AnthropicLLM, LLMError  # noqa: E402
from eu_law_nli.registry import list_documents, load_document  # noqa: E402

st.set_page_config(page_title="EU LexRef", page_icon="⚖️")


def password_gate() -> None:
    """Ask for the shared test password when NLI_APP_PASSWORD is set (on
    Streamlit Community Cloud, in the app's Secrets). Stops the page until
    the right password is entered; nothing else loads or calls the model."""
    expected = config.APP_PASSWORD
    if not expected or st.session_state.get("authenticated"):
        return
    # In the browser's language, from the saved translations (no model call here).
    t = Localiser(None).strings((st.context.locale or "en").split("-")[0].lower())
    st.title(t["page_title"])
    st.markdown(f"**{t['page_subtitle']}**")
    st.write(t["login_intro"])
    with st.form("login", border=True, enter_to_submit=True):
        # "current-password" tells browsers this is an existing password, so phones
        # do not offer to create (and save) a new one.
        entered = st.text_input(t["login_password"], type="password",
                                autocomplete="current-password")
        # Users accept the disclaimer once here; answers then carry it only once.
        acknowledged = st.checkbox(t["login_acknowledge"])
        submitted = st.form_submit_button(t["login_button"], type="primary",
                                          use_container_width=True)
    st.caption(t["login_help"])
    if submitted:
        if not (entered and hmac.compare_digest(entered.encode(), expected.encode())):
            st.error(t["login_error"])
        elif not acknowledged:
            st.error(t["login_acknowledge_missing"])
        else:
            st.session_state["authenticated"] = True
            st.rerun()
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
def get_engine(key: tuple, library_keys: tuple, wording: str) -> Engine:
    # "wording" changes whenever the fixed texts in i18n.py change, so new code
    # deployed without a restart never meets an engine built with the old texts.
    document, version, _ = key
    return Engine(AnthropicLLM(), document, version, library=get_library(library_keys))


WORDING_VERSION = str(hash(tuple(sorted(i18n.STRINGS.items()))))


def browser_language() -> str:
    return (st.context.locale or "en").split("-")[0].lower()


documents = list_documents()
all_docs = [load_document(d) for d in documents]
default_id = config.DEFAULT_DOCUMENT if config.DEFAULT_DOCUMENT in documents else documents[0]
previous: Session | None = st.session_state.get("session")
page_language = previous.language if previous else browser_language()
# The engine is not loaded yet, so these labels come from the saved translations.
ui = Localiser(None, extra=document_strings(all_docs)).strings(page_language)

# Compare each act's stored text with the published one, in background
# threads: at start, then every LIVE_CHECK_HOURS. Nobody waits for it.
for d in documents:
    live.start_background(d)


def select_key(language: str) -> str:
    # One dropdown per page language: Streamlit keeps showing an option's text
    # from when the dropdown was first drawn, so a new language needs a new
    # dropdown for the act names to appear in that language.
    return f"doc_select_{language}"


if st.session_state.pop("go_home", False) and previous is not None:
    # After an answer from another act, the web app returns to the home act
    # (config.DEFAULT_DOCUMENT), keeping the conversation on screen.
    st.session_state[select_key(page_language)] = default_id
    st.session_state["carry_session"] = previous
    st.session_state["carry_shown"] = "returned_home"

for stale in [k for k in st.session_state if str(k).startswith("doc_select_")
              and k != select_key(page_language)]:
    del st.session_state[stale]  # an earlier language's dropdown may hold an old choice

with st.sidebar:
    # Shown even with one act, so more acts can be added to documents/ later.
    current = st.session_state.get("document", default_id)
    doc_id = st.selectbox(ui["legal_act_label"], documents, key=select_key(page_language),
                          index=documents.index(current),
                          format_func=lambda d: ui.get(f"{d}.name", d))

try:
    if live.is_checking(doc_id) and st.session_state.get("engine_key", ("",))[0] == doc_id:
        engine_key = st.session_state["engine_key"]  # keep the current text while files change
    else:
        engine_key = text_key(doc_id)
    st.session_state["engine_key"] = engine_key
    library_keys = tuple(text_key(d) for d in documents)
    engine = get_engine(engine_key, library_keys, WORDING_VERSION)
except (LLMError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()

if st.session_state.get("document") != doc_id or "session" not in st.session_state:
    st.session_state["document"] = doc_id
    carried = st.session_state.pop("carry_session", None)
    if carried:  # switching acts keeps the language and the recent conversation
        session = continue_in(carried, engine.doc.short_name, "web")
    else:
        session = Session(interface="web")
        # Until the user writes, show the page in the browser's language when the
        # act is loaded in it (its wording is then translated already).
        browser = browser_language()
        if browser.upper() in engine.corpora:
            session.language = browser
    st.session_state["session"] = session
    carried_notice = st.session_state.pop("carry_shown", None)
    if carried_notice:  # switching acts keeps the conversation on screen
        st.session_state["act_notice"] = carried_notice
    else:
        st.session_state["shown"] = []
session: Session = st.session_state["session"]
shown: list[tuple[str, object]] = st.session_state.setdefault("shown", [])
# English as a fallback: after an update without a restart, a cached engine may
# not yet know a newly added text.
strings = {**i18n.STRINGS, **engine.strings(session)}
notice = st.session_state.pop("act_notice", None)
if notice:
    shown.append(("notice", strings.get(notice, i18n.STRINGS.get(notice, "{document}")).format(
        document=strings.get(f"{doc_id}.name", engine.doc.short_name))))


def show_reply(reply: Reply, latest: bool) -> None:
    """One assistant reply: summary and explanation first, the exact wording
    collapsed, then notes, a suggested act and the small print (the
    disclaimer on the first reply only)."""
    if reply.summary:
        st.markdown(f"**{reply.summary}**")
    if reply.details:
        st.markdown(reply.details)
    if reply.wording:
        label = strings["show_wording"] if reply.quotes else strings["closest_heading"]
        with st.expander(label):
            st.markdown(reply.wording)
    for note in reply.notes:
        st.caption(note)
    suggestion = getattr(reply, "suggestion", None)
    if suggestion and latest and suggestion["document"] != doc_id:
        name = strings.get(f"{suggestion['document']}.name", suggestion["name"])
        st.button(strings["switch_button"].format(document=name), key="switch",
                  on_click=_switch, args=(suggestion["document"], suggestion["question"]))
    if reply.followup:
        st.markdown(reply.followup)
    st.caption("  \n".join(x for x in (reply.disclaimer, reply.source) if x))


def _ask(text: str) -> None:
    st.session_state["queued"] = text


def _switch(document: str, question: str) -> None:
    """Change the selected act and ask the same question there."""
    st.session_state[select_key(session.language)] = document
    st.session_state["queued"] = question
    st.session_state["carry_session"] = session
    st.session_state["carry_shown"] = "switched_to"


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
        for key in ("session", "shown", "queued"):
            st.session_state.pop(key, None)
        st.rerun()

st.title(strings["page_title"])
st.markdown(f"**{strings['page_subtitle']}**")
intro = strings["page_intro"]
if config.USAGE_LOG:
    intro += " " + strings["audit_notice"].format(days=config.USAGE_RETENTION_DAYS)
st.caption(intro)

for index, (role, item) in enumerate(shown):
    if role == "notice":
        st.info(item)
        continue
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
    # In italics, to set them apart from the grey introduction above.
    st.markdown(f"*{strings['examples_heading']}*")
    for i, example in enumerate(examples):
        st.button(f"*{example}*", key=f"example_{i}", on_click=_ask, args=(example,))

typed = st.chat_input(strings["input_placeholder"])
prompt = typed or st.session_state.pop("queued", None)

away_from_home = config.RETURN_HOME and doc_id != default_id

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
    st.session_state["go_home"] = away_from_home  # back to the home act for the next question
    st.rerun()  # redraw with the final layout, translated into the conversation's language
