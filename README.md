# EU law reference assistant (prototype)

A natural language interface for a library of EU acts and other reference
material: currently the European Electronic Communications Code (Directive
(EU) 2018/1972), the General Data Protection Regulation and the BEREC
Regulation. Users choose an act and ask in any language; the assistant answers
in that language, quotes the act word for word, links each quotation to the
official text, and points to another act in the library when that one covers
the question better. Claude runs through Anthropic's API, Amazon Bedrock or
Microsoft Foundry.

## Set up

Python 3.10 or later.

```
pip install -r requirements.txt
cp .env.example .env            # then put your Anthropic API key in .env
python -m eu_law_nli.ingest eecc --lang ALL   # all 24 official languages
python -m eu_law_nli.ingest gdpr --lang ALL
python -m eu_law_nli.ingest berec --lang ALL
streamlit run app.py            # web chat
python cli.py                   # or chat in the terminal
```

The ingest step downloads each act and stores it in `data/`. With
`--lang ALL` every official EU language is built (or name some:
`--lang EN ET FR DE`). A question asked in one of those languages is answered
with quotations and links from that language version; any other language gets
English quotations, and the reply says so.

EUR-Lex often turns away automated downloads. The ingest then fetches the same
text from the Publications Office repository (publications.europa.eu), which
allows scripts. Only if both fail does it print the page to open in your
browser and the path to save it to (`data/raw/<CELEX>_<LANG>.html`, as "Web
page, HTML only"); run the same command again afterwards.

The model is `claude-sonnet-5-5` by default. Change it with `NLI_MODEL` in
`.env` (current ids: https://docs.claude.com/en/docs/about-claude/models), and
how much it thinks with `NLI_EFFORT` (`low`, the default, up to `max`).

## What it does on each message

| Step | What happens | Where |
|---|---|---|
| Analyse | Detects the language, the intent (question, situation, yes/no, greeting) and whether the topic is within the selected act | model call, `prompts.py` |
| Route | Searches every act in the library and notes another act that matches clearly better | local, `library.py` |
| Retrieve | Finds the relevant passages of the selected act by keyword search | local, `retriever.py` |
| Answer | Writes the reply from those passages only and selects quotations | model call |
| Verify | Keeps a quotation only if it is found word for word in the stored text; the stored wording is what gets displayed | local, `engine.py` |
| Finish | Adds references, the offer to compare a situation, the source line and the disclaimer | local, `engine.py` |

Behaviour to expect:

- **Answers.** Lead with one or two plain sentences a non-lawyer understands,
  naming the Code ("Yes. Under the European Electronic Communications Code,
  ..."), then a short explanation citing the articles (under 120 words, never
  "the passages"). In the web app the explanation appears word by word as it
  is written, and the exact quotations sit under "Show the exact wording from
  the Code", collapsed. Disclaimer and source follow in small print.
- **Limits and requests for new topics.** When a question, or part of one,
  depends on national law or another EU act, a short note says the assistant
  cannot review it and asks the user to check that law first. The first time
  this happens in a conversation, the user is asked once, with Yes/No buttons,
  whether such subjects may be recorded for the author. A yes covers the rest
  of the conversation; a no, or carrying on without answering, means nothing
  is recorded and the question is not asked again. Recorded subjects go to
  `data/feedback/out_of_scope.jsonl`: the question and each topic with its
  source (national law, other EU act). Read the log in the terminal with
  `python -m eu_law_nli.feedback`; visitors never see it.
- **The page.** Title, notices, input box, buttons and small print follow the
  language of the conversation (before the first question, the browser's
  language, for the 24 EU languages). The empty page offers two example
  questions. Errors are explained in plain words; the technical detail goes
  to the usage log only.
- **Situations.** Answers that do not ask the consent question offer to
  compare a situation. When the user
  describes one, the reply lists the relevant provisions, how they may apply,
  and which facts are missing. It does not give a conclusion.
- **Disclaimer.** Added by the code to every reply, including errors, so the
  model cannot omit it. The wording is in `eu_law_nli/i18n.py`.
- **Languages.** Fixed wording is translated once per language by the model
  and saved in `data/i18n/<code>.json`. Have the disclaimer translations
  reviewed and correct them in those files. When an English text in `i18n.py`
  changes, only that text is translated again; corrections to the others stay.

## Amendments

Each act has a registry file in `documents/`. The assistant always uses the
latest version listed in `documents/eecc.json`, currently the EUR-Lex
consolidated text of 18 October 2024. Every quotation and link comes from that
version. The 2018 Official Journal text stays in the registry for comparison
only.

Consolidated texts leave out the recitals (the explanatory preamble), so the
assistant does not quote them. To include them, set `recitals_from_original`
to `true` in the registry file; those quotations then link to the 2018 text.

### Live check against EUR-Lex

The assistant answers from a stored copy of the text, but checks it against
the live EU version in a background thread: when the web app or the terminal
chat starts, then every 24 hours while it runs. Nobody waits for the check;
answers come from the stored copy meanwhile. Each check:

1. It asks the Publications Office database (the source of EUR-Lex) which
   consolidated versions of the act exist. EUR-Lex's own pages turn scripts
   away; this public service does not.
2. If a newer version exists, it downloads it in every language already
   loaded, adds it to `documents/eecc.json` and uses it from then on. Earlier
   versions stay under `data/corpus/eecc/<version>/` (`--version 2018-12-17`).
3. Otherwise it downloads the current English text and compares it, provision
   by provision, with the stored copy, rebuilding if the wording differs.

The check takes a few seconds; adopting a new version takes about half a
minute for 24 languages. When a newer or corrected text is ready, the next
question is answered from it, with no restart; while files are being
written, answers keep using the previous text. The source line of every reply says when the text
was last checked ("Checked against the live text on EUR-Lex on 2026-10-07
08:36 UTC"). If the EU servers cannot be reached, the assistant keeps using
the stored copy and the source line says the check failed. The web app
sidebar shows the last check.

```
python -m eu_law_nli.ingest eecc --sync            # run the check by hand
python -m eu_law_nli.ingest eecc --check-updates   # only list newer versions
```

Settings in `.env`: `NLI_LIVE_CHECK=off` to stop checking, and
`NLI_LIVE_CHECK_HOURS` (default 24). The last result is kept in
`data/live/eecc.json`.

Note that consolidated texts are documentation tools without legal effect;
only the Official Journal text is authentic. Every reply states which version
it used.

## The library: several acts, one question

The library currently holds three acts: the European Electronic Communications
Code (`eecc`), the General Data Protection Regulation (`gdpr`) and the BEREC
Regulation (`berec`). The user picks one under "Legal act"; answers come from
that act only.

Every question is also searched across all the acts in the library
(`eu_law_nli/library.py`: one shared keyword index, so scores compare across
acts). When another act matches clearly better, the reply names it and the
provisions that matched, and offers a "Switch to … and ask there" button that
changes the act and asks the same question again. In the terminal chat, type
`/switch` (or `/switch <id>`; `/acts` lists the library).

- If the selected act does not deal with the question, another act is
  suggested when it matches better than the selected one. Nothing is recorded
  for the author then, because the library already covers the subject.
- After an answer, another act is suggested only when it scores at least
  `NLI_SUGGEST_RATIO` (1.5) times higher, so borderline questions stay quiet.
- `NLI_SUGGEST_MIN_SCORE` (10) is the least score worth suggesting. Both values
  were calibrated on questions about the three acts; recheck them as the
  library grows.

## Adding reference material

Each document has a registry file in `documents/` naming its source type
(`eu_law_nli/sources.py`). Everything else (search, quotation checks, live
updates, translations, the web app) works the same for every source.

**An EU act from EUR-Lex** (any directive or regulation with ELI ids, which all
recent acts have):

```
python -m eu_law_nli.ingest gdpr --add-eurlex 32016R0679 \
    --short-name "General Data Protection Regulation" \
    --citation "Regulation (EU) 2016/679" \
    --scope "Protection of natural persons with regard to the processing of personal data: ..."
python -m eu_law_nli.ingest gdpr --lang ALL
```

The first command writes `documents/gdpr.json` with the Official Journal text
and the latest consolidated version, found automatically; the second builds
all 24 languages. Then add two `examples` (start-page questions, in English)
to the file. The `scope` sentence matters: the assistant uses it to decide
whether a question belongs to the act.

**Any other reference point** (guidelines, national laws, internal notes):
copy `documents/templates/web-source.example.json` to `documents/<id>.json`,
set `"source": "web"` and give each version either a `url` (a web page) or a
`path` (a local file), with `format` html, markdown or text. The text is split
into sections at its headings, and quotations are checked against it like any
act. Run `python -m eu_law_nli.ingest <id>`.

**Updates** are picked up by the live check for every source: EUR-Lex acts
adopt newer consolidated versions; web pages and files are downloaded or read
again and rebuilt when their text changed. To add a new kind of source (PDF,
a national legal database), write a class with the same five methods as
`WebSource` in `sources.py` and add it to `SOURCES`.

## Running on Amazon Bedrock or Microsoft Foundry

Claude can run in your own cloud account instead of Anthropic's API. Set
`NLI_PROVIDER` in `.env`; nothing else in the code changes, because all three
offer the same Messages API with the features used here (JSON schema output,
effort, streaming).

| Provider | Settings | Model |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `NLI_MODEL` |
| `bedrock` | AWS credentials from your profile, environment variables or IAM role; `NLI_AWS_REGION` | `anthropic.` + `NLI_MODEL`, or `NLI_BEDROCK_MODEL` for an inference profile such as `eu.anthropic.claude-sonnet-5-5` |
| `foundry` | `ANTHROPIC_FOUNDRY_RESOURCE`, and `ANTHROPIC_FOUNDRY_API_KEY` or `NLI_FOUNDRY_AUTH=entra` (Entra ID sign-in, `pip install azure-identity`) | `NLI_FOUNDRY_DEPLOYMENT` |

The Bedrock and Foundry set-up is covered by tests with stand-in clients but
has not been run against a live account. With Entra ID, confirm the token
scope (`NLI_FOUNDRY_TOKEN_SCOPE`) with your Azure administrator. The usage log
records the provider; its cost estimate uses Anthropic's list prices, which
cloud billing may differ from.

## Usage log

Every message is logged to `data/usage/usage.jsonl`, one line per turn: the
time, a random conversation id, web or terminal, the user's language, the
message as typed, the kind of reply (answer, situation, out of scope, error
...), the articles quoted, how many quotations passed or failed the
word-for-word check, subjects flagged as outside the Code, response time,
tokens and model. The assistant's reply is not stored.

Entries are deleted automatically after 90 days: each time a turn is logged or
the log is read, expired entries are removed. Both the web app and the
terminal chat show this notice:

> Audit log: during this trial, the questions are kept for 90 days for
> testing and improvement, then deleted automatically. Nothing that identifies
> you is recorded: no name, account, IP address or device details. Please do
> not include personal or confidential details in your questions.

The notice holds only while the code keeps it true: the conversation id is
random, and no IP address, request headers or device details are read. If
you add logins or analytics, update the notice.

```
python -m eu_law_nli.usage             # summary: volumes, languages, articles, errors, cost
python -m eu_law_nli.usage --days 7    # the last 7 days only
python -m eu_law_nli.usage --list 20   # also the 20 most recent questions
```

Settings in `.env`:
`NLI_USAGE_RETENTION_DAYS` (default 90) and `NLI_USAGE_LOG=off` to stop
logging. The cost estimate uses the prices in `config.PRICES`.

The log holds what users typed, which may include personal data despite the
notice. Before wider use, name a controller and a purpose in a privacy notice,
keep `data/` out of version control (it is in `.gitignore`) and limit who can
read the folder.

## Tests

```
python -m unittest discover -s tests -v
```

The tests run offline against two shortened EUR-Lex pages and a scripted
stand-in for the model.

## Limits of this prototype

- Each answer comes from one act. Questions spanning several acts are pointed
  to the best-matching act, not answered across acts together. National
  implementing laws, case law and guidelines can be added as web sources.
- Retrieval is keyword based. It works for a few acts; for a large library or
  weaker phrasing, add an embedding search beside `BM25Retriever`.
- Each user works in a separate conversation. Replies are not stored; the
  questions are kept in the usage log for 90 days (see above).
- Messages, including any situation a user describes, are sent to the model
  provider. Tell users not to include personal or confidential details, and
  check your data-processing terms before wider use.
- One user and one document are assumed per session; there is no login.
