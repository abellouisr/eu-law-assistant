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
import html
import logging
import os
import time
import uuid

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

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# Navy for the assistant, a light slate blue for the user: told apart at a
# glance, and no red or orange, which read as alerts.
AVATARS = {"user": os.path.join(ASSETS, "avatar_user.svg"),
           "assistant": os.path.join(ASSETS, "avatar_assistant.svg")}

# Layout touches the theme settings cannot express: a slim header, the summary
# as a key-finding box and the cited articles as small linked labels.
st.markdown("""<style>
[data-testid="stMainBlockContainer"] { padding-top: 4rem; }
.lexref-brand { display: flex; align-items: center; gap: .65rem; margin: 0 0 .75rem; }
.lexref-mark { flex: none; width: 2.1rem; height: 2.1rem; border-radius: 6px;
  background: #1f3a5f; color: #fff; display: flex; align-items: center;
  justify-content: center; font: 700 1.3rem Georgia, "Times New Roman", serif; }
.lexref-name { font-size: 1.45rem; font-weight: 700; color: #1f3a5f; line-height: 1.1; }
.lexref-tagline { font-size: .9rem; color: #5b6573; line-height: 1.2; }
.lexref-summary { border-left: 3px solid #1f3a5f; background: #eef2f7;
  padding: .6rem .9rem; border-radius: 0 6px 6px 0; margin: 0 0 .8rem; }
.lexref-cites { display: flex; flex-wrap: wrap; gap: .4rem; margin: .1rem 0 .6rem; }
.lexref-cite { font-size: .8rem; padding: .12rem .6rem; border: 1px solid #c5d0de;
  border-radius: 999px; background: #f3f6fa; color: #1f3a5f !important;
  text-decoration: none !important; white-space: nowrap; }
.lexref-cite:hover { background: #e3eaf3; border-color: #1f3a5f; }
/* Section titles in the sidebar ("Legal act", "Recent conversations"): one style. */
.lexref-section { font-size: .78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .05em; color: #5b6573; margin: .6rem 0 .3rem; }
/* Recent conversations: a left-aligned list; the open one in navy, not greyed out. */
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] { justify-content: flex-start;
  text-align: left; width: 100%; padding: .15rem .25rem; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p { text-align: left; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:disabled { color: #1f3a5f;
  font-weight: 600; opacity: 1; }
</style>""", unsafe_allow_html=True)


def section_title(text: str) -> None:
    """A sidebar section heading, styled the same everywhere."""
    st.markdown(f'<div class="lexref-section">{html.escape(text)}</div>',
                unsafe_allow_html=True)


def brand_header(t: dict) -> None:
    """The product name and tagline in one slim line."""
    st.markdown(
        f'<div class="lexref-brand"><div class="lexref-mark">§</div><div>'
        f'<div class="lexref-name">{html.escape(t["page_title"])}</div>'
        f'<div class="lexref-tagline">{html.escape(t["page_subtitle"])}</div></div></div>',
        unsafe_allow_html=True)


def password_gate() -> None:
    """Ask for the shared test password when NLI_APP_PASSWORD is set (on
    Streamlit Community Cloud, in the app's Secrets). Stops the page until
    the right password is entered; nothing else loads or calls the model."""
    expected = config.APP_PASSWORD
    if not expected or st.session_state.get("authenticated"):
        return
    # In the browser's language, from the saved translations (no model call here).
    t = Localiser(None).strings((st.context.locale or "en").split("-")[0].lower())
    brand_header(t)
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

def _new_conversation() -> None:
    for key in ("session", "shown", "queued"):
        st.session_state.pop(key, None)
    st.session_state["conv_id"] = uuid.uuid4().hex  # the old one stays in the list


def _open_conversation(conv_id: str) -> None:
    """Bring back an earlier conversation from this visit, with its act and language."""
    saved = st.session_state["conversations"][conv_id]
    for key in ("queued", "go_home", "carry_session", "carry_shown", "act_notice"):
        st.session_state.pop(key, None)
    st.session_state.update(conv_id=conv_id, session=saved["session"], shown=saved["shown"],
                            document=saved["document"])
    st.session_state[select_key(saved["session"].language)] = saved["document"]


with st.sidebar:
    st.button(ui["new_conversation"], icon=":material/add:", use_container_width=True,
              on_click=_new_conversation)
    # Shown even with one act, so more acts can be added to documents/ later.
    current = st.session_state.get("document", default_id)
    section_title(ui["legal_act_label"])
    # The visible heading is the section title above; the label stays for screen readers.
    doc_id = st.selectbox(ui["legal_act_label"], documents, key=select_key(page_language),
                          index=documents.index(current), label_visibility="collapsed",
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
        st.session_state["conv_id"] = uuid.uuid4().hex  # a fresh conversation
session: Session = st.session_state["session"]
shown: list[tuple[str, object]] = st.session_state.setdefault("shown", [])
# English as a fallback: after an update without a restart, a cached engine may
# not yet know a newly added text.
strings = {**i18n.STRINGS, **engine.strings(session)}
notice = st.session_state.pop("act_notice", None)
if notice:
    shown.append(("notice", strings.get(notice, i18n.STRINGS.get(notice, "{document}")).format(
        document=strings.get(f"{doc_id}.name", engine.doc.short_name))))

# Recent conversations of this visit, for the sidebar. Kept in the browser
# session's memory only: they go when the page is closed, and are not logged.
# The stored "shown" is the same list the page appends to, so it stays current.
MAX_CONVERSATIONS = 10
conversations: dict[str, dict] = st.session_state.setdefault("conversations", {})
conv_id = st.session_state.setdefault("conv_id", uuid.uuid4().hex)
saved = conversations.get(conv_id)
if saved is None or saved["shown"] is not shown or len(shown) != saved["length"]:
    conversations[conv_id] = {"session": session, "shown": shown, "document": doc_id,
                              "length": len(shown), "updated": time.time()}
else:  # the act may have changed, with a new session object
    saved.update(session=session, document=doc_id)
for old in sorted(conversations, key=lambda c: conversations[c]["updated"])[:-MAX_CONVERSATIONS]:
    del conversations[old]


def conversation_title(entries: list) -> str:
    """The first question asked, shortened: the conversation's name in the list."""
    first = next((item for role, item in entries if role == "user"), "")
    first = " ".join(str(first).split())
    return first if len(first) <= 42 else first[:40].rstrip() + "…"


def show_reply(reply: Reply, latest: bool) -> None:
    """One assistant reply: summary and explanation first, the exact wording
    collapsed, then notes, a suggested act and the small print (the
    disclaimer on the first reply only)."""
    if reply.summary:
        st.markdown(f'<div class="lexref-summary">{html.escape(reply.summary)}</div>',
                    unsafe_allow_html=True)
    if reply.details:
        st.markdown(reply.details)
    cited = citation_labels(reply)
    if cited:
        st.markdown(f'<div class="lexref-cites">{cited}</div>', unsafe_allow_html=True)
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


def citation_labels(reply: Reply) -> str:
    """The articles an answer quotes, once each, as labels linking to EUR-Lex."""
    labels: dict[str, str] = {}
    for quote in getattr(reply, "quotes", []):
        if quote.reference in labels:
            continue
        text = html.escape(quote.reference)
        labels[quote.reference] = (
            f'<a class="lexref-cite" href="{html.escape(quote.url, quote=True)}" '
            f'target="_blank" rel="noopener">{text}</a>' if quote.url
            else f'<span class="lexref-cite">{text}</span>')
    return "".join(labels.values())


def _ask(text: str) -> None:
    st.session_state["queued"] = text


def _switch(document: str, question: str) -> None:
    """Change the selected act and ask the same question there."""
    st.session_state[select_key(session.language)] = document
    st.session_state["queued"] = question
    st.session_state["carry_session"] = session
    st.session_state["carry_shown"] = "switched_to"


with st.sidebar, st.container(border=True):  # the text in use, grouped
    version = engine.version
    st.markdown(f"**{engine.doc.citation}**")
    version_text = strings.get(f"version_line_{version.kind}",
                               strings["version_line_document"]).format(
        date=format_date(version.date))
    state = live.status(doc_id)
    if config.LIVE_CHECK and state.get("ok") and state.get("version") == version.id:
        version_text += ", " + strings["latest_note"].format(
            checked=format_date(state.get("last_ok_at") or state["checked_at"]))
    st.caption(version_text)
    lang = session.language.upper() if session.language.upper() in engine.corpora \
        else engine.reference.lang
    official = sources.link(engine.doc, version, lang)
    if official:
        st.link_button(strings["open_eurlex"], official, icon=":material/open_in_new:",
                       use_container_width=True)

with st.sidebar:
    listed = [(c, conversations[c]) for c in sorted(
        conversations, key=lambda c: conversations[c]["updated"], reverse=True)
        if conversation_title(conversations[c]["shown"])]
    if listed:
        section_title(strings["recent_conversations"])
        for cid, saved in listed:
            current_one = cid == conv_id
            st.button(conversation_title(saved["shown"]), key=f"conv_{cid}", type="tertiary",
                      icon=":material/chat:" if current_one else ":material/history:",
                      disabled=current_one, on_click=_open_conversation, args=(cid,),
                      help=strings.get(f"{saved['document']}.name"))

brand_header(strings)
intro = strings["page_intro"]
if config.USAGE_LOG:
    intro += " " + strings["audit_notice"].format(days=config.USAGE_RETENTION_DAYS)
if shown:  # once the conversation starts, the introduction moves to the sidebar
    st.sidebar.caption(intro)
else:
    st.caption(intro)

for index, (role, item) in enumerate(shown):
    if role == "notice":
        st.info(item)
        continue
    with st.chat_message(role, avatar=AVATARS[role]):
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
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
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
