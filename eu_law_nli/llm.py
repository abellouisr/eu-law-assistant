"""The language-model boundary.

The engine only needs one operation: "given these instructions and this
input, return JSON that fits this schema". Any provider that can do that can
be plugged in by writing a class with the same ``json`` method. When
``on_text`` is given, it is called with the JSON text received so far, so the
web app can show the answer while it is being written.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Callable, Protocol

from . import config


class LLMError(RuntimeError):
    pass


class LLMBusy(LLMError):
    """The provider is overloaded or rate-limited: worth trying again shortly."""


class LLM(Protocol):
    def json(self, system: str, user: str, schema: dict, name: str,
             on_text: Callable[[str], None] | None = None) -> dict: ...


def make_client(provider: str | None = None, api_key: str | None = None):
    """The Claude client for the configured provider, and the model id to use there.

    anthropic  Anthropic's API. Needs ANTHROPIC_API_KEY.
    bedrock    Amazon Bedrock, in your AWS account. Credentials come from the usual
               AWS chain (profile, environment, instance or task role); the region
               from NLI_AWS_REGION or AWS_REGION. Model ids carry an "anthropic."
               prefix; set NLI_BEDROCK_MODEL for an inference profile id.
    foundry    Microsoft Foundry, in your Azure subscription. Needs
               ANTHROPIC_FOUNDRY_RESOURCE (or ANTHROPIC_FOUNDRY_BASE_URL) and either
               ANTHROPIC_FOUNDRY_API_KEY or NLI_FOUNDRY_AUTH=entra (Entra ID through
               azure-identity). The model is the deployment name, NLI_FOUNDRY_DEPLOYMENT.

    All three expose the same Messages API, so nothing else in the code changes.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise LLMError("Install the SDK first: pip install anthropic") from exc
    provider = (provider or config.PROVIDER).lower()
    if provider == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Put it in the .env file.")
        return anthropic.Anthropic(api_key=key), config.MODEL
    if provider == "bedrock":
        model = config.BEDROCK_MODEL or (
            config.MODEL if "anthropic." in config.MODEL else f"anthropic.{config.MODEL}")
        try:
            return anthropic.AnthropicBedrockMantle(aws_region=config.AWS_REGION or None), model
        except Exception as exc:
            raise LLMError(f"Could not set up Amazon Bedrock: {exc}") from exc
    if provider == "foundry":
        token_provider = None
        if config.FOUNDRY_AUTH == "entra":
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as exc:
                raise LLMError("Entra ID sign-in needs: pip install azure-identity") from exc
            token_provider = get_bearer_token_provider(DefaultAzureCredential(),
                                                       config.FOUNDRY_TOKEN_SCOPE)
        try:
            client = anthropic.AnthropicFoundry(api_key=api_key,
                                                azure_ad_token_provider=token_provider)
        except Exception as exc:
            raise LLMError(f"Could not set up Microsoft Foundry: {exc}") from exc
        return client, config.FOUNDRY_DEPLOYMENT or config.MODEL
    raise LLMError(f"Unknown NLI_PROVIDER '{provider}'. Use anthropic, bedrock or foundry.")


class AnthropicLLM:
    """Claude through Anthropic's API, Amazon Bedrock or Microsoft Foundry
    (NLI_PROVIDER). Structured output uses the JSON schema output format, so
    the reply is guaranteed to parse and fit."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 max_tokens: int | None = None, provider: str | None = None) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client, default_model = make_client(provider, api_key)
        self.provider = (provider or config.PROVIDER).lower()
        self.model = model or default_model
        self.max_tokens = max_tokens or config.MAX_TOKENS
        # Token counts per thread: the web app serves each visitor in its own
        # thread, so one visitor's turn never picks up another's usage.
        self._usage = threading.local()

    def tokens(self) -> tuple[int, int]:
        """Input and output tokens used so far by the calling thread."""
        return getattr(self._usage, "input", 0), getattr(self._usage, "output", 0)

    def json(self, system: str, user: str, schema: dict, name: str,
             on_text: Callable[[str], None] | None = None) -> dict:
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": config.EFFORT,
                "format": {"type": "json_schema", "schema": _closed(schema)},
            },
        )
        anthropic = self._anthropic
        try:
            if on_text is None:
                response = self._client.messages.create(**request)
            else:
                with self._client.messages.stream(**request) as stream:
                    received = ""
                    for piece in stream.text_stream:
                        received += piece
                        on_text(received)
                    response = stream.get_final_message()
        except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            raise LLMBusy(f"The model service is busy: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 529:
                raise LLMBusy(f"The model service is overloaded: {exc}") from exc
            raise LLMError(f"The model request failed: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMBusy(f"Could not reach the model service: {exc}") from exc
        used = getattr(response, "usage", None)
        if used is not None:
            self._usage.input = getattr(self._usage, "input", 0) + (used.input_tokens or 0)
            self._usage.output = getattr(self._usage, "output", 0) + (used.output_tokens or 0)
        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"The model ran out of output space for '{name}'. "
                           "Raise NLI_MAX_TOKENS.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned no structured result.") from exc


def _closed(schema: dict) -> dict:
    """Copy of the schema with ``additionalProperties: false`` on every
    object, which the JSON schema output format requires."""
    if not isinstance(schema, dict):
        return schema
    out = {k: (_closed(v) if isinstance(v, dict) else
               [_closed(i) for i in v] if isinstance(v, list) else v)
           for k, v in schema.items()}
    if "properties" in out:
        out["properties"] = {k: _closed(v) for k, v in schema["properties"].items()}
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    return out


class ScriptedLLM:
    """Stand-in used by the tests: answers each call from a function or a queue."""

    def __init__(self, responder=None, queue: list[dict] | None = None) -> None:
        self.responder = responder
        self.queue = list(queue or [])
        self.calls: list[dict] = []

    def json(self, system: str, user: str, schema: dict, name: str,
             on_text: Callable[[str], None] | None = None) -> dict:
        self.calls.append({"system": system, "user": user, "name": name})
        if self.responder is not None:
            result = self.responder(name, system, user)
        elif self.queue:
            result = self.queue.pop(0)
        else:
            raise LLMError(f"ScriptedLLM has no response left for '{name}'")
        if on_text is not None:
            on_text(json.dumps(result, ensure_ascii=False))
        return result
