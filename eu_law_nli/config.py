"""Paths and settings. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from .env without overriding real environment variables."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

DOCUMENTS_DIR = Path(os.environ.get("NLI_DOCUMENTS_DIR", ROOT / "documents"))
DATA_DIR = Path(os.environ.get("NLI_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CORPUS_DIR = DATA_DIR / "corpus"
I18N_DIR = DATA_DIR / "i18n"
FEEDBACK_FILE = DATA_DIR / "feedback" / "out_of_scope.jsonl"
LIVE_DIR = DATA_DIR / "live"

# Check the live EU text at every start and then every LIVE_CHECK_HOURS.
LIVE_CHECK = os.environ.get("NLI_LIVE_CHECK", "on").lower() not in {"off", "0", "false", "no"}
LIVE_CHECK_HOURS = float(os.environ.get("NLI_LIVE_CHECK_HOURS", "24"))

# Usage log: one line per turn, including the user's message. Entries older
# than the retention period are deleted automatically.
USAGE_LOG = os.environ.get("NLI_USAGE_LOG", "on").lower() not in {"off", "0", "false", "no"}
USAGE_FILE = DATA_DIR / "usage" / "usage.jsonl"
USAGE_RETENTION_DAYS = int(os.environ.get("NLI_USAGE_RETENTION_DAYS", "90"))
# A private Google Sheet that also receives the usage and topics logs, so they
# survive restarts of a hosted copy (see sheets.py). Off unless an id is set.
# Credentials: GCP_SERVICE_ACCOUNT_JSON (the key's JSON text, e.g. in Streamlit
# Secrets) or NLI_GSHEET_CREDENTIALS_FILE (a path to the key file).
GSHEET_ID = os.environ.get("NLI_GSHEET_ID", "").strip()
GSHEET_CREDENTIALS_FILE = os.environ.get("NLI_GSHEET_CREDENTIALS_FILE", "").strip()
# USD per million input / output tokens, for the cost estimate in the summary.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00),
          "claude-haiku-4-5": (1.00, 5.00)}

# Where Claude runs: anthropic (Anthropic's API), bedrock (Amazon Bedrock) or
# foundry (Microsoft Foundry). See llm.make_client for each one's settings.
PROVIDER = os.environ.get("NLI_PROVIDER", "anthropic").lower()
AWS_REGION = os.environ.get("NLI_AWS_REGION") or os.environ.get("AWS_REGION", "")
BEDROCK_MODEL = os.environ.get("NLI_BEDROCK_MODEL", "")  # e.g. an EU inference profile id
FOUNDRY_DEPLOYMENT = os.environ.get("NLI_FOUNDRY_DEPLOYMENT", "")
FOUNDRY_AUTH = os.environ.get("NLI_FOUNDRY_AUTH", "key").lower()  # key | entra
# Entra ID token scope for Foundry; confirm it with your Azure administrator.
FOUNDRY_TOKEN_SCOPE = os.environ.get("NLI_FOUNDRY_TOKEN_SCOPE",
                                     "https://cognitiveservices.azure.com/.default")

# Set NLI_MODEL to a current model id from https://docs.claude.com/en/docs/about-claude/models
MODEL = os.environ.get("NLI_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.environ.get("NLI_MAX_TOKENS", "16000"))
# How hard the model thinks: low, medium, high, xhigh or max
EFFORT = os.environ.get("NLI_EFFORT", "low")

DEFAULT_DOCUMENT = os.environ.get("NLI_DOCUMENT", "eecc")
# The web app returns to the default ("home") act after answering in another act.
RETURN_HOME = os.environ.get("NLI_RETURN_HOME", "on").lower() not in {"off", "0", "false", "no"}
# Shared password for test users of the web app; empty means no password screen.
APP_PASSWORD = os.environ.get("NLI_APP_PASSWORD", "")
TOP_K = int(os.environ.get("NLI_TOP_K", "8"))
MAX_CONTEXT_CHARS = int(os.environ.get("NLI_MAX_CONTEXT_CHARS", "28000"))
HISTORY_TURNS = int(os.environ.get("NLI_HISTORY_TURNS", "6"))
# Suggest switching to another act when it matches the question at least this
# well (keyword score) and, after an answer, this many times better than the
# selected act. Calibrated on EECC, GDPR and BEREC questions.
SUGGEST_MIN_SCORE = float(os.environ.get("NLI_SUGGEST_MIN_SCORE", "10"))
SUGGEST_RATIO = float(os.environ.get("NLI_SUGGEST_RATIO", "1.5"))

EURLEX_BASE = "https://eur-lex.europa.eu/legal-content"


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temporary file and swap it in, so a reader in another
    thread (the background live check runs beside the chat) never sees a
    half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def eurlex_url(celex: str, lang: str, anchor: str = "") -> str:
    url = f"{EURLEX_BASE}/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}"
    return f"{url}#{anchor}" if anchor else url
