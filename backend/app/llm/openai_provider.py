"""
EchoMind OpenAI-Compatible LLM Gateway Provider — Phase 4.0

Connects EchoMind to any standard OpenAI Chat Completions endpoint
(OpenAI, vLLM, Ollama, LM Studio, AI Kosh, OpenCode).

Performs live HTTP request execution, latency measurement, logging, SSE streaming token parsing,
and strict error handling via EchoMindException.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.exceptions import EchoMindException
from app.core.logging import logger
from app.llm.providers.base import BaseLLMProvider


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    OpenAI-Compatible LLM Gateway Provider for EchoMind.

    Responsibilities:
      • Build OpenAI Chat Completion requests (headers, payloads).
      • Perform live HTTP execution via httpx AsyncClient.
      • Measure request latency and log provider, model, status, latency.
      • Parse SSE streamed chunks and non-streamed JSON responses.
      • Raise EchoMindException for network/endpoint failures without silent fallback.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str = "mock-gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> None:
        self.base_url = (base_url or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key
        self.model = model or "mock-gpt-4o"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream

    def build_headers(self) -> dict[str, str]:
        """Builds HTTP headers required for OpenAI-compatible endpoint calls."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_payload(
        self,
        query: str,
        context_prompt: str,
        stream: bool | None = None,
    ) -> dict[str, Any]:
        """Builds OpenAI Chat Completion API specification payload."""
        content = context_prompt if context_prompt else query
        use_stream = self.stream if stream is None else stream
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are EchoMind AI assistant, an expert software developer and digital memory agent."},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": use_stream,
        }

    def parse_response(self, response_json: dict[str, Any]) -> str:
        """Parses standard non-streamed OpenAI Chat Completion JSON response."""
        try:
            choices = response_json.get("choices", [])
            if not choices:
                raise EchoMindException("Invalid API response: no choices returned.", status_code=502)
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if content is None:
                return ""
            return str(content).strip()
        except Exception as exc:
            if isinstance(exc, EchoMindException):
                raise
            raise EchoMindException(f"Failed to parse OpenAI API response: {exc}", status_code=502)

    def parse_stream_chunk(self, chunk_line: str) -> str | None:
        """
        Parses Server-Sent Event (SSE) stream chunk line.
        Line format: 'data: {"choices": [{"delta": {"content": "..."}}]}'
        Returns token delta string or None.
        """
        line = chunk_line.strip()
        if not line or not line.startswith("data:"):
            return None
        payload_str = line[5:].strip()
        if payload_str == "[DONE]":
            return None
        try:
            data = json.loads(payload_str)
            choices = data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content")
        except Exception:
            return None
        return None

    async def generate_answer(
        self,
        query: str,
        context_prompt: str,
        results: list[Any] = None,
    ) -> str:
        """
        Executes Chat Completion request against configured base_url endpoint using httpx.
        Measures latency and logs provider, model, status, and latency.
        Raises EchoMindException for network or API failures. Never falls back silently.
        """
        headers = self.build_headers()
        payload = self.build_payload(query, context_prompt, stream=False)
        url = f"{self.base_url}/chat/completions"

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                logger.info(
                    f"LLM Gateway Request: provider=openai, model='{self.model}', "
                    f"status={res.status_code}, latency={latency_ms:.2f}ms"
                )

                if res.status_code != 200:
                    logger.error(f"OpenAI Endpoint Error ({res.status_code}): {res.text}")
                    raise EchoMindException(
                        f"OpenAI Endpoint Error (HTTP {res.status_code}): {res.text}",
                        status_code=res.status_code,
                    )

                response_json = res.json()
                return self.parse_response(response_json)

        except httpx.HTTPError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"OpenAI Endpoint Network Failure ({url}) after {latency_ms:.2f}ms: {err}")
            raise EchoMindException(
                f"Failed to connect to OpenAI-compatible endpoint '{url}': {err}",
                status_code=503,
            ) from err
        except Exception as exc:
            if isinstance(exc, EchoMindException):
                raise
            raise EchoMindException(
                f"Unexpected error communicating with endpoint '{url}': {exc}",
                status_code=500,
            ) from exc

    async def generate_answer_stream(
        self,
        query: str,
        context_prompt: str,
        results: list[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams SSE tokens from OpenAI-compatible endpoint.
        """
        headers = self.build_headers()
        payload = self.build_payload(query, context_prompt, stream=True)
        url = f"{self.base_url}/chat/completions"

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as res:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    logger.info(
                        f"LLM Gateway Stream Connected: provider=openai, model='{self.model}', "
                        f"status={res.status_code}, connect_latency={latency_ms:.2f}ms"
                    )

                    if res.status_code != 200:
                        error_body = await res.aread()
                        raise EchoMindException(
                            f"OpenAI Streaming Endpoint Error (HTTP {res.status_code}): {error_body.decode()}",
                            status_code=res.status_code,
                        )

                    async for line in res.aiter_lines():
                        token = self.parse_stream_chunk(line)
                        if token:
                            yield token

        except httpx.HTTPError as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"OpenAI Streaming Network Failure ({url}) after {latency_ms:.2f}ms: {err}")
            raise EchoMindException(
                f"Failed to connect to OpenAI-compatible streaming endpoint '{url}': {err}",
                status_code=503,
            ) from err
