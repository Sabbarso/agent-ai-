"""OpenAI-compatible LLM client (Groq, Gemini, OpenAI, Ollama...).

Features: timeout, exponential-backoff retries, on-disk cache keyed by
hash(model + messages + params), call counters, and `structured()` output validated by
Pydantic with a corrective retry when the JSON is invalid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from rpe.config import Config, EnvSettings, load_env
from rpe.llm import prompts

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, str]]
CompletionFn = Callable[..., str]
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMError(RuntimeError):
    """The LLM call failed after all retries."""


class LLMNotConfiguredError(LLMError):
    """LLM_BASE_URL / LLM_MODEL missing from the environment."""


class StructuredOutputError(LLMError):
    """The model never produced JSON valid for the requested schema."""


class TransientLLMError(LLMError):
    """Retryable error (used by custom completion functions and tests)."""


@dataclass
class LLMStats:
    n_calls: int = 0
    n_cache_hits: int = 0
    n_retries: int = 0
    n_parse_failures: int = 0

    def __str__(self) -> str:
        return (
            f"LLM calls={self.n_calls} cache_hits={self.n_cache_hits} "
            f"retries={self.n_retries} parse_failures={self.n_parse_failures}"
        )


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply (tolerates code fences and surrounding text)."""
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    if not starts:
        raise ValueError("No JSON object found in the reply")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[min(starts) :])
    return obj


def _retryable_errors() -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [TransientLLMError, TimeoutError, ConnectionError]
    try:
        import openai

        errors += [
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ]
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors)


class LLMClient:
    """Chat + structured outputs over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_parse_retries: int = 2,
        json_mode: bool = True,
        cache_dir: Path | None = None,
        completion_fn: CompletionFn | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_parse_retries = max_parse_retries
        self.json_mode = json_mode
        self.cache_dir = cache_dir
        self.completion_fn = completion_fn
        self.sleep = sleep
        self.stats = LLMStats()
        self._client: Any = None
        self._retryable = _retryable_errors()

    # ------------------------------------------------------------ factories

    @classmethod
    def from_config(cls, cfg: Config, env: EnvSettings | None = None) -> LLMClient:
        env = env or load_env()
        if not env.llm_configured:
            raise LLMNotConfiguredError(
                "LLM not configured: set LLM_BASE_URL, LLM_MODEL (and LLM_API_KEY) in .env"
            )
        c = cfg.llm
        return cls(
            model=env.llm_model or "",
            base_url=env.llm_base_url,
            api_key=env.llm_api_key.get_secret_value() if env.llm_api_key else None,
            timeout=c.timeout,
            max_retries=c.max_retries,
            backoff_seconds=c.backoff_seconds,
            temperature=c.temperature,
            max_tokens=c.max_tokens,
            max_parse_retries=c.max_parse_retries,
            json_mode=c.json_mode,
            cache_dir=cfg.path(c.cache_dir),
        )

    # ---------------------------------------------------------------- cache

    def cache_key(self, messages: Messages, temperature: float, json_mode: bool) -> str:
        payload = {"model": self.model, "messages": messages, "t": temperature, "json": json_mode}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    def _cache_get(self, key: str) -> str | None:
        p = self._cache_path(key)
        if p is None or not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))["response"]

    def _cache_put(self, key: str, response: str) -> None:
        p = self._cache_path(key)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"model": self.model, "response": response}), encoding="utf-8")
        tmp.replace(p)

    def _cache_drop(self, key: str) -> None:
        p = self._cache_path(key)
        if p is not None and p.exists():
            p.unlink()

    # ------------------------------------------------------------ transport

    def _openai(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self._api_key or "not-needed",
                timeout=self.timeout,
                max_retries=0,  # retries are handled here, with logging
            )
        return self._client

    def _raw_complete(
        self, messages: Messages, temperature: float, max_tokens: int, json_mode: bool
    ) -> str:
        if self.completion_fn is not None:
            return self.completion_fn(
                messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        import openai

        kwargs: dict[str, Any] = {}
        if json_mode and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._openai().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except openai.BadRequestError as exc:
            if "response_format" in kwargs and "response_format" in str(exc).lower():
                logger.warning("Provider rejected response_format; disabling JSON mode")
                self.json_mode = False
                return self._raw_complete(messages, temperature, max_tokens, json_mode)
            raise
        return resp.choices[0].message.content or ""

    def _complete(
        self, messages: Messages, temperature: float, max_tokens: int, json_mode: bool
    ) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                self.stats.n_calls += 1
                logger.debug("LLM call #%d (%s)", self.stats.n_calls, self.model)
                return self._raw_complete(messages, temperature, max_tokens, json_mode)
            except self._retryable as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"LLM call failed after {attempt + 1} attempts: {exc}") from exc
                delay = self.backoff_seconds * (2**attempt)
                self.stats.n_retries += 1
                logger.warning("LLM error (%s), retry %d in %.1fs", exc, attempt + 1, delay)
                self.sleep(delay)
        raise LLMError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ API

    @staticmethod
    def build_messages(prompt: str | Messages, system: str | None = None) -> Messages:
        if isinstance(prompt, list):
            return ([{"role": "system", "content": system}] if system else []) + prompt
        msgs: Messages = [{"role": "system", "content": system}] if system else []
        return msgs + [{"role": "user", "content": prompt}]

    def chat(
        self,
        prompt: str | Messages,
        system: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        use_cache: bool = True,
    ) -> str:
        """Plain completion (cached by input hash)."""
        messages = self.build_messages(prompt, system)
        t = self.temperature if temperature is None else temperature
        key = self.cache_key(messages, t, json_mode)
        if use_cache and (hit := self._cache_get(key)) is not None:
            self.stats.n_cache_hits += 1
            return hit
        out = self._complete(messages, t, max_tokens or self.max_tokens, json_mode)
        if use_cache:
            self._cache_put(key, out)
        return out

    def structured(
        self,
        prompt: str | Messages,
        schema: type[T],
        system: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Completion parsed and validated as `schema`; retries with the validation error."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        sys_prompt = "\n\n".join(
            s for s in (system, prompts.JSON_OUTPUT_INSTRUCTIONS.format(schema=schema_json)) if s
        )
        messages = self.build_messages(prompt, sys_prompt)
        t = self.temperature if temperature is None else temperature
        last_error = ""
        for _ in range(self.max_parse_retries + 1):
            text = self.chat(messages, temperature=t, max_tokens=max_tokens, json_mode=True)
            try:
                return schema.model_validate(parse_json(text))
            except (ValueError, ValidationError) as exc:
                self.stats.n_parse_failures += 1
                last_error = str(exc)[:800]
                logger.warning("Invalid structured output for %s: %s", schema.__name__, last_error)
                self._cache_drop(self.cache_key(messages, t, True))
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": prompts.FIX_JSON.format(error=last_error)},
                ]
        raise StructuredOutputError(f"No valid {schema.__name__} after retries: {last_error}")


def get_llm(cfg: Config) -> LLMClient | None:
    """LLM client from config, or `None` when the LLM is not configured."""
    try:
        return LLMClient.from_config(cfg)
    except LLMNotConfiguredError:
        return None
