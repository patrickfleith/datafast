"""Capability-aware LLM providers for Datafast."""

from __future__ import annotations

import copy
import os
import random
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel

import litellm
from litellm import exceptions as litellm_exceptions

from datafast.llm.capabilities import resolve_capabilities
from datafast.llm.types import (
    BatchMode,
    ContentPart,
    EndpointMode,
    Message,
    Messages,
    Modality,
    NormalizedRequest,
    NormalizedResponse,
    RetryPolicy,
    StructuredOutputMode,
    ServedModelCapabilities,
    ServedModelConfig,
    UnsupportedParamsPolicy,
)
from datafast.tracing import (
    build_trace_metadata,
    load_env_once,
    maybe_configure_langfuse_tracing,
)


T = TypeVar("T", bound=BaseModel)

JSON_INSTRUCTIONS = (
    "\nReturn only valid JSON. Do not include markdown fences. Use double quotes "
    "for keys and string values, escape internal newlines, and avoid trailing commas."
)
LITELLM_SUPPRESS_DEBUG_ENV = "DATAFAST_LITELLM_SUPPRESS_DEBUG_INFO"


def _configure_litellm_debug_output() -> None:
    """Suppress LiteLLM provider help text unless explicitly opted out."""
    setting = os.getenv(LITELLM_SUPPRESS_DEBUG_ENV, "1").strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return
    litellm.suppress_debug_info = True


class ServedModel:
    """One provider-and-model pair resolved to LiteLLM request adapters."""

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        *,
        litellm_route: str,
        env_key_name: str | None,
        endpoint_mode: str | EndpointMode = EndpointMode.AUTO,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        rpm_limit: int | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
        api_base_url: str | None = None,
        api_base: str | None = None,
        retry_limit: int | None = None,
        retry_policy: RetryPolicy | None = None,
        unsupported_params: str | UnsupportedParamsPolicy = UnsupportedParamsPolicy.WARN,
        provider_params: dict[str, Any] | None = None,
        max_concurrent: int = 4,
        capabilities: ServedModelCapabilities | None = None,
        **extra_provider_params: Any,
    ) -> None:
        if max_completion_tokens is None and max_tokens is not None:
            max_completion_tokens = max_tokens
        if api_base_url is None:
            api_base_url = api_base

        merged_provider_params = dict(provider_params or {})
        merged_provider_params.update(extra_provider_params)

        if retry_policy is None:
            retry_policy = RetryPolicy(
                max_retries=retry_limit if retry_limit is not None else 3
            )

        unsupported_policy = _coerce_unsupported_policy(unsupported_params)

        self.config = ServedModelConfig(
            provider_id=provider_id,
            model_id=model_id,
            litellm_route=litellm_route,
            env_key_name=env_key_name,
            endpoint_mode=_coerce_endpoint_mode(endpoint_mode),
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            max_completion_tokens=max_completion_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            rpm_limit=rpm_limit,
            timeout=timeout,
            api_key=api_key,
            api_base_url=api_base_url,
            retry_policy=retry_policy,
            unsupported_params=unsupported_policy,
            provider_params=merged_provider_params,
            max_concurrent=max_concurrent,
        )
        self.capabilities = resolve_capabilities(
            provider_id,
            model_id,
            api_base_url=api_base_url,
            explicit=capabilities,
        )
        self.endpoint_mode = self._resolve_endpoint_mode(self.config.endpoint_mode)

        self.provider_id = provider_id
        self.model_id = model_id
        self.env_key_name = env_key_name
        env_api_key = os.getenv(env_key_name) if env_key_name else None
        # `or None` so an empty env var is treated as unset, not sent as "".
        self.api_key = api_key or env_api_key or None
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.rpm_limit = rpm_limit
        self.timeout = timeout
        self.unsupported_params = unsupported_policy.value

        self._request_timestamps: list[float] = []
        self._rate_lock = Lock()
        self._sleep = time.sleep
        self._configured_common_params = {
            name
            for name, value in {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "max_completion_tokens": max_completion_tokens,
                "thinking": thinking,
                "reasoning_effort": reasoning_effort,
                "timeout": timeout,
            }.items()
            if value is not None
        }

        _configure_litellm_debug_output()
        load_env_once()
        maybe_configure_langfuse_tracing(load_env=False)
        logger.info(
            "Initialized {} | Model: {} | Endpoint: {}",
            self.provider_id,
            self.model_id,
            self.endpoint_mode.value,
        )

    def generate(
        self,
        prompt: str | list[str] | None = None,
        messages: Messages | list[Messages] | None = None,
        response_format: type[T] | None = None,
        metadata: dict[str, Any] | None = None,
        previous_response_id: str | None = None,
    ) -> str | list[str] | T | list[T]:
        """Generate a single response or ordered batch of responses."""
        requests, single_input = self._normalize_inputs(
            prompt=prompt,
            messages=messages,
            metadata=metadata,
            previous_response_id=previous_response_id,
            response_format=response_format,
        )
        results = self._generate_requests(requests, response_format=response_format)
        if single_input:
            return results[0]
        return results

    def generate_batch(
        self,
        messages: list[Messages],
        *,
        response_format: type[T] | None = None,
        metadata: list[dict[str, Any] | None] | dict[str, Any] | None = None,
        previous_response_ids: list[str | None] | None = None,
    ) -> list[str] | list[T]:
        """Generate an ordered batch from pre-built message lists."""
        if not messages:
            return []

        metadata_items = _normalize_metadata(metadata, len(messages))
        previous_ids = (
            previous_response_ids
            if previous_response_ids is not None
            else [None] * len(messages)
        )
        if len(previous_ids) != len(messages):
            raise ValueError("previous_response_ids length must match messages length")

        requests = [
            NormalizedRequest(
                messages=self._prepare_messages(
                    item,
                    response_format=response_format,
                ),
                metadata=metadata_items[index],
                previous_response_id=previous_ids[index],
            )
            for index, item in enumerate(messages)
        ]
        return self._generate_requests(requests, response_format=response_format)

    def generate_response(
        self,
        prompt: str | list[str] | None = None,
        messages: Messages | list[Messages] | None = None,
        metadata: dict[str, Any] | None = None,
        previous_response_id: str | None = None,
    ) -> NormalizedResponse | list[NormalizedResponse]:
        """Generate response metadata, including LiteLLM reasoning fields when present."""
        requests, single_input = self._normalize_inputs(
            prompt=prompt,
            messages=messages,
            metadata=metadata,
            previous_response_id=previous_response_id,
            response_format=None,
        )
        responses = self._execute_normalized(
            requests,
            response_format=None,
        )
        if single_input:
            return responses[0]
        return responses

    def generate_batch_response(
        self,
        messages: list[Messages],
        *,
        metadata: list[dict[str, Any] | None] | dict[str, Any] | None = None,
        previous_response_ids: list[str | None] | None = None,
    ) -> list[NormalizedResponse]:
        """Generate ordered batch responses with metadata preserved."""
        if not messages:
            return []

        metadata_items = _normalize_metadata(metadata, len(messages))
        previous_ids = (
            previous_response_ids
            if previous_response_ids is not None
            else [None] * len(messages)
        )
        if len(previous_ids) != len(messages):
            raise ValueError("previous_response_ids length must match messages length")

        requests = [
            NormalizedRequest(
                messages=self._prepare_messages(item, response_format=None),
                metadata=metadata_items[index],
                previous_response_id=previous_ids[index],
            )
            for index, item in enumerate(messages)
        ]
        return self._execute_normalized(requests, response_format=None)

    def _generate_requests(
        self,
        requests: list[NormalizedRequest],
        *,
        response_format: type[T] | None,
    ) -> list[str] | list[T]:
        responses = self._execute_normalized(
            requests,
            response_format=response_format,
        )
        return [
            self._parse_response(response, response_format=response_format)
            for response in responses
        ]

    def _execute_normalized(
        self,
        requests: list[NormalizedRequest],
        *,
        response_format: type[T] | None,
    ) -> list[NormalizedResponse]:
        """Run requests with the shared error contract: ValueError passes
        through (validation/policy), everything else is logged and wrapped in
        RuntimeError."""
        try:
            return self._generate_normalized_responses(
                requests,
                response_format=response_format,
            )
        except ValueError:
            raise
        except Exception as exc:
            error_trace = traceback.format_exc()
            logger.error(
                "Generation failed | Provider: {} | Model: {} | Error: {}",
                self.provider_id,
                self.model_id,
                exc,
            )
            raise RuntimeError(
                f"Error generating response with {self.provider_id}:\n{error_trace}"
            ) from exc

    def _generate_normalized_responses(
        self,
        requests: list[NormalizedRequest],
        *,
        response_format: type[T] | None,
    ) -> list[NormalizedResponse]:
        if not requests:
            return []

        if len(requests) == 1:
            return [self._execute_single(requests[0], response_format=response_format)]

        if (
            self.endpoint_mode == EndpointMode.CHAT
            and self.capabilities.batch_mode == BatchMode.LITELLM_BATCH
        ):
            return self._execute_litellm_batch(
                requests,
                response_format=response_format,
            )

        warnings.warn(
            (
                f"{self.provider_id}/{self.model_id} does not expose native "
                "batching for this endpoint. Falling back to bounded "
                "parallel single requests."
            ),
            UserWarning,
            stacklevel=2,
        )
        with ThreadPoolExecutor(
            max_workers=max(1, min(self.config.max_concurrent, len(requests)))
        ) as executor:
            responses = list(
                executor.map(
                    lambda request: self._execute_single(
                        request,
                        response_format=response_format,
                    ),
                    requests,
                )
            )
        return responses

    def _execute_single(
        self,
        request: NormalizedRequest,
        *,
        response_format: type[T] | None,
    ) -> NormalizedResponse:
        if self.endpoint_mode == EndpointMode.RESPONSES:
            params = self._build_responses_params(request, response_format)
            response = self._call_litellm(
                litellm.responses,
                params,
                request_count=1,
            )
            return NormalizedResponse(
                text=_extract_responses_text(response),
                raw=response,
                reasoning_content=_extract_responses_reasoning(response),
                images=_extract_responses_images(response),
                audio=_extract_responses_audio(response),
                output_items=_extract_responses_output_items(response),
            )

        params = self._build_chat_params(request, response_format)
        response = self._call_litellm(
            litellm.completion,
            params,
            request_count=1,
        )
        return NormalizedResponse(
            text=_extract_chat_text(response),
            raw=response,
            reasoning_content=_extract_chat_reasoning_content(response),
            thinking_blocks=_extract_chat_thinking_blocks(response),
            images=_extract_chat_images(response),
            audio=_extract_chat_audio(response),
        )

    def _execute_litellm_batch(
        self,
        requests: list[NormalizedRequest],
        *,
        response_format: type[T] | None,
    ) -> list[NormalizedResponse]:
        if any(request.previous_response_id is not None for request in requests):
            # batch_completion shares one param set across items, so
            # per-request response ids cannot be carried on this path.
            self._handle_unsupported_param("previous_response_id")

        params = self._build_chat_params(
            NormalizedRequest(
                messages=[],
                metadata=_combine_batch_metadata(requests),
            ),
            response_format,
        )
        params["messages"] = [request.messages for request in requests]
        params["max_workers"] = max(
            1, min(self.config.max_concurrent, len(requests))
        )
        response = self._call_litellm(
            litellm.batch_completion,
            params,
            request_count=len(requests),
        )
        if not isinstance(response, list):
            response = list(response)

        normalized: list[NormalizedResponse] = []
        for index, item in enumerate(response):
            if isinstance(item, Exception):
                # batch_completion returns per-item exceptions instead of
                # raising, which bypasses the retry policy; re-run the item
                # through the single-request path so retries/backoff apply.
                normalized.append(
                    self._retry_failed_batch_item(
                        requests[index],
                        index=index,
                        error=item,
                        response_format=response_format,
                    )
                )
                continue
            normalized.append(
                NormalizedResponse(
                    text=_extract_chat_text(item),
                    raw=item,
                    reasoning_content=_extract_chat_reasoning_content(item),
                    thinking_blocks=_extract_chat_thinking_blocks(item),
                    images=_extract_chat_images(item),
                    audio=_extract_chat_audio(item),
                )
            )
        return normalized

    def _retry_failed_batch_item(
        self,
        request: NormalizedRequest,
        *,
        index: int,
        error: Exception,
        response_format: type[T] | None,
    ) -> NormalizedResponse:
        if not (
            _is_retryable_error(error)
            or (
                self.config.unsupported_params != UnsupportedParamsPolicy.FAIL
                and _is_unsupported_params_error(error)
            )
        ):
            raise RuntimeError(f"Batch item {index} failed: {error}") from error
        try:
            return self._execute_single(request, response_format=response_format)
        except Exception as retry_error:
            raise RuntimeError(
                f"Batch item {index} failed after retries: {retry_error}"
            ) from retry_error

    def _build_chat_params(
        self,
        request: NormalizedRequest,
        response_format: type[T] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._get_model_string(),
            "messages": request.messages,
            "metadata": self._build_request_metadata(request.metadata),
        }
        if request.previous_response_id is not None:
            # previous_response_id is a Responses-API concept; chat completions
            # endpoints reject it regardless of what the served model supports.
            self._handle_unsupported_param("previous_response_id")
        self._add_transport_params(params, endpoint=EndpointMode.CHAT)
        self._add_common_generation_params(params, endpoint=EndpointMode.CHAT)
        self._add_chat_structured_output(params, response_format)
        params.update(self.config.provider_params)
        self._apply_reasoning_allowlist(params)
        return _without_none(params)

    def _build_responses_params(
        self,
        request: NormalizedRequest,
        response_format: type[T] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._get_model_string(),
            "input": _to_responses_input(request.messages),
            # For the Responses API, "metadata" is a wire parameter forwarded
            # to the provider (OpenAI requires string values); trace metadata
            # goes through LiteLLM's logging-only litellm_metadata instead.
            "litellm_metadata": self._build_request_metadata(request.metadata),
        }
        if request.previous_response_id is not None:
            self._add_supported_param(
                params,
                "previous_response_id",
                request.previous_response_id,
                endpoint=EndpointMode.RESPONSES,
            )
        self._add_transport_params(params, endpoint=EndpointMode.RESPONSES)
        self._add_common_generation_params(params, endpoint=EndpointMode.RESPONSES)
        self._add_responses_structured_output(params, response_format)
        params.update(self.config.provider_params)
        return _without_none(params)

    def _add_common_generation_params(
        self,
        params: dict[str, Any],
        *,
        endpoint: EndpointMode,
    ) -> None:
        if self._temperature_allowed():
            self._add_supported_param(
                params,
                "temperature",
                self.config.temperature,
                endpoint=endpoint,
            )
        elif "temperature" in self._configured_common_params:
            self._handle_unsupported_param(
                "temperature", detail="while reasoning is enabled"
            )

        self._add_supported_param(
            params,
            "top_p",
            self.config.top_p,
            endpoint=endpoint,
        )
        self._add_supported_param(
            params,
            "frequency_penalty",
            self.config.frequency_penalty,
            endpoint=endpoint,
        )

        token_param = (
            "max_output_tokens"
            if endpoint == EndpointMode.RESPONSES
            else "max_completion_tokens"
        )
        self._add_supported_param(
            params,
            "max_completion_tokens",
            self.config.max_completion_tokens,
            endpoint=endpoint,
            param_name=token_param,
        )

        if self.config.thinking is False:
            if self.config.reasoning_summary is not None:
                self._handle_unsupported_param(
                    "reasoning_summary", detail="while reasoning is disabled"
                )
            if self.capabilities.reasoning_always_on:
                raise ValueError(
                    f"{self.provider_id}/{self.model_id} always reasons, so "
                    "thinking=False cannot be honoured. Pass reasoning_effort "
                    "with the lowest level it accepts instead"
                    + self._supported_efforts_hint()
                )
            off_param = self.capabilities.reasoning_off_param
            if off_param is not None:
                name, value = off_param
                # The off value is an effort like any other, so it needs the
                # same Responses-shaped wrapper the "on" path applies below.
                if name == "reasoning_effort" and endpoint == EndpointMode.RESPONSES:
                    params["reasoning"] = {"effort": value}
                else:
                    params[name] = value
            return

        effort = self._resolve_reasoning_effort()
        summary = self._resolve_reasoning_summary(endpoint)

        if endpoint == EndpointMode.RESPONSES and (
            effort is not None or summary is not None
        ):
            reasoning = {"effort": effort, "summary": summary}
            self._add_supported_param(
                params,
                "reasoning_effort",
                {k: v for k, v in reasoning.items() if v is not None},
                endpoint=endpoint,
                param_name="reasoning",
            )
            return

        self._add_supported_param(
            params,
            "reasoning_effort",
            effort,
            endpoint=endpoint,
        )

    def _temperature_allowed(self) -> bool:
        """False where the served model rejects a temperature with reasoning on."""
        if not self.capabilities.reasoning_locks_temperature:
            return True
        if self.config.thinking is False:
            return True
        return self._resolve_reasoning_effort() is None

    def _supported_efforts_hint(self) -> str:
        """The accepted levels, for error messages, or '' where any is allowed."""
        supported = self.capabilities.reasoning_efforts
        if not supported:
            return "."
        return f": {', '.join(sorted(supported))}."

    def _resolve_reasoning_summary(self, endpoint: EndpointMode) -> str | None:
        """Resolve the reasoning summary to ask for, or None to omit it.

        A summary is a Responses-only concept: it rides inside the `reasoning`
        object, which chat endpoints have no field for, whatever a profile
        that serves both endpoints declares.
        """
        summary = self.config.reasoning_summary
        if summary is None:
            return None
        if (
            endpoint != EndpointMode.RESPONSES
            or "reasoning_summary" not in self.capabilities.supported_params
        ):
            self._handle_unsupported_param("reasoning_summary")
            return None
        return summary

    def _resolve_reasoning_effort(self) -> str | None:
        """Resolve the reasoning_effort value to send, or None to omit it.

        thinking=True means the served model's own "on" level rather than a
        fixed one: Mistral's reasoning models accept only 'high' and 'none' and
        reject 'low' with a 400.
        """
        effort = self.config.reasoning_effort
        if effort is None:
            if self.config.thinking is not True:
                return None
            effort = self.capabilities.reasoning_effort_on

        supported = self.capabilities.reasoning_efforts
        if supported is not None and effort not in supported:
            raise ValueError(
                f"reasoning_effort '{effort}' is not supported by "
                f"{self.provider_id}/{self.model_id}. Supported values: "
                f"{', '.join(sorted(supported))}."
            )
        return effort

    def _apply_reasoning_allowlist(self, params: dict[str, Any]) -> None:
        """Force reasoning_effort past LiteLLM's per-model param filter.

        Some served models (e.g. Mistral's mistral-medium/small) accept reasoning_effort
        server-side, but the installed LiteLLM only recognises it for a subset of
        models and would otherwise drop it. allowed_openai_params tells LiteLLM to
        forward the parameter anyway.
        """
        if not self.capabilities.reasoning_requires_allowlist:
            return
        if "reasoning_effort" not in params:
            return
        allowed = list(params.get("allowed_openai_params") or [])
        if "reasoning_effort" not in allowed:
            allowed.append("reasoning_effort")
            params["allowed_openai_params"] = allowed

    def _add_chat_structured_output(
        self,
        params: dict[str, Any],
        response_format: type[T] | None,
    ) -> None:
        if response_format is None:
            return

        mode = self.capabilities.structured_output
        if mode == StructuredOutputMode.JSON_SCHEMA:
            params["response_format"] = response_format
        elif mode == StructuredOutputMode.JSON_OBJECT:
            params["response_format"] = {"type": "json_object"}
        elif mode == StructuredOutputMode.PROMPTED_JSON:
            warnings.warn(
                (
                    f"{self.provider_id}/{self.model_id} has no declared native "
                    "schema support. Using prompted JSON plus Pydantic validation."
                ),
                UserWarning,
                stacklevel=3,
            )
        else:
            raise ValueError(
                f"{self.provider_id}/{self.model_id} does not support structured output"
            )

    def _add_responses_structured_output(
        self,
        params: dict[str, Any],
        response_format: type[T] | None,
    ) -> None:
        if response_format is None:
            return

        if self.capabilities.structured_output != StructuredOutputMode.JSON_SCHEMA:
            raise ValueError(
                f"{self.provider_id}/{self.model_id} does not support native "
                "Responses structured output"
            )
        params["text_format"] = response_format

    def _add_transport_params(
        self,
        params: dict[str, Any],
        *,
        endpoint: EndpointMode,
    ) -> None:
        if self.config.timeout is not None:
            self._add_supported_param(
                params,
                "timeout",
                self.config.timeout,
                endpoint=endpoint,
            )
        if self.api_base_url is not None:
            params["api_base"] = self.api_base_url
        # no_api_key only waives auth for self-hosted served models (custom base
        # URL); hosted endpoints still require a key even if the resolved
        # capability profile is a keyless local one.
        api_key_optional = self.capabilities.no_api_key and (
            self.api_base_url is not None or self.env_key_name is None
        )
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.env_key_name and not api_key_optional:
            env_key = os.getenv(self.env_key_name)
            if env_key:
                params["api_key"] = env_key
            else:
                raise ValueError(
                    f"{self.env_key_name} environment variable not set. "
                    "Set it or provide api_key when initializing the served model."
                )

    def _add_supported_param(
        self,
        params: dict[str, Any],
        source_name: str,
        value: Any,
        *,
        endpoint: EndpointMode,
        param_name: str | None = None,
    ) -> None:
        if value is None:
            return

        if source_name not in self.capabilities.supported_params:
            if (
                source_name in self._configured_common_params
                or source_name == "previous_response_id"
                or (
                    source_name == "reasoning_effort"
                    and self.config.thinking is True
                )
            ):
                self._handle_unsupported_param(source_name)
            return

        if source_name == "reasoning_effort" and not self.capabilities.supports_reasoning:
            self._handle_unsupported_param(source_name)
            return

        if endpoint == EndpointMode.RESPONSES and not self.capabilities.supports_endpoint(
            EndpointMode.RESPONSES
        ):
            self._handle_unsupported_param(source_name)
            return

        params[param_name or source_name] = value

    def _handle_unsupported_param(self, name: str, detail: str | None = None) -> None:
        target = f"resolved served model {self.provider_id}/{self.model_id}"
        if detail:
            target = f"{target} {detail}"
        message = f"Parameter '{name}' is not supported by {target} and will be omitted."
        if self.config.unsupported_params == UnsupportedParamsPolicy.FAIL:
            raise ValueError(message)
        if self.config.unsupported_params == UnsupportedParamsPolicy.WARN:
            warnings.warn(message, UserWarning, stacklevel=3)

    def _normalize_inputs(
        self,
        *,
        prompt: str | list[str] | None,
        messages: Messages | list[Messages] | None,
        metadata: dict[str, Any] | None,
        previous_response_id: str | None,
        response_format: type[T] | None,
    ) -> tuple[list[NormalizedRequest], bool]:
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        if prompt is not None and messages is not None:
            raise ValueError("Provide either prompt or messages, not both")

        single_input = False
        batch_messages: list[Messages]

        if prompt is not None:
            if isinstance(prompt, str):
                batch_messages = [[{"role": "user", "content": prompt}]]
                single_input = True
            elif isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
                if not prompt:
                    raise ValueError("prompt list cannot be empty")
                batch_messages = [
                    [{"role": "user", "content": item}]
                    for item in prompt
                ]
            else:
                raise ValueError("prompt must be a string or list of strings")
        elif _is_single_messages(messages):
            batch_messages = [messages]  # type: ignore[list-item]
            single_input = True
        elif _is_batch_messages(messages):
            batch_messages = messages  # type: ignore[assignment]
            if not batch_messages:
                raise ValueError("messages cannot be empty")
        else:
            raise ValueError("Invalid messages format")

        return (
            [
                NormalizedRequest(
                    messages=self._prepare_messages(
                        item,
                        response_format=response_format,
                    ),
                    metadata=metadata,
                    previous_response_id=previous_response_id,
                )
                for item in batch_messages
            ],
            single_input,
        )

    def _prepare_messages(
        self,
        messages: Messages,
        *,
        response_format: type[T] | None,
    ) -> Messages:
        if not messages:
            raise ValueError("messages cannot be empty")

        normalized = [
            _normalize_message(
                message,
                include_media_uuid=self.capabilities.supports_media_uuid,
            )
            for message in copy.deepcopy(messages)
        ]
        self._validate_modalities(normalized)

        if response_format is not None and self.capabilities.structured_output in {
            StructuredOutputMode.JSON_OBJECT,
            StructuredOutputMode.PROMPTED_JSON,
        }:
            _append_json_instructions(normalized)

        return normalized

    def _validate_modalities(self, messages: Messages) -> None:
        supported = self.capabilities.modalities
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                modality = _modality_for_part(part)
                if modality not in supported:
                    raise ValueError(
                        f"Modality '{modality.value}' is not supported by "
                        f"{self.provider_id}/{self.model_id}"
                    )
                if modality == Modality.FILE:
                    self._validate_file_carrier(part)

    def _validate_file_carrier(self, part: dict[str, Any]) -> None:
        """Reject a file part the served model could not accept.

        Declaring Modality.FILE says files work, not how they arrive. Where only an
        uploaded id will do, inline data reaches the provider as a malformed part and
        comes back as an opaque schema error, so it is worth catching here.
        """
        if not self.capabilities.files_require_file_id:
            return
        if part.get("file", {}).get("file_id"):
            return
        raise ValueError(
            f"{self.provider_id}/{self.model_id} accepts a file only as an id from "
            "its own upload API. Upload it with the served model's upload_file(), "
            "then pass the returned id as the file part's 'url' — "
            "ContentPart(type='file', url=<file_id>) — rather than inline 'data'."
        )

    def _resolve_endpoint_mode(self, endpoint_mode: EndpointMode) -> EndpointMode:
        if endpoint_mode == EndpointMode.AUTO:
            return self.capabilities.default_endpoint_mode
        if not self.capabilities.supports_endpoint(endpoint_mode):
            raise ValueError(
                f"{self.provider_id}/{self.model_id} does not support "
                f"endpoint_mode='{endpoint_mode.value}'"
            )
        return endpoint_mode

    def _call_litellm(self, func, params: dict[str, Any], *, request_count: int) -> Any:
        try:
            return self._call_with_retries(
                lambda: func(**params),
                request_count=request_count,
            )
        except Exception as exc:
            if not self._should_retry_with_drop_params(exc, params):
                raise

            retry_params = dict(params)
            retry_params["drop_params"] = True
            if self.config.unsupported_params == UnsupportedParamsPolicy.WARN:
                warnings.warn(
                    (
                        "LiteLLM rejected one or more request parameters as "
                        "unsupported. Retrying once with drop_params=True because "
                        f"unsupported_params='{self.config.unsupported_params.value}'."
                    ),
                    UserWarning,
                    stacklevel=3,
                )
            return self._call_with_retries(
                lambda: func(**retry_params),
                request_count=request_count,
            )

    def _should_retry_with_drop_params(
        self,
        exc: Exception,
        params: dict[str, Any],
    ) -> bool:
        if self.config.unsupported_params == UnsupportedParamsPolicy.FAIL:
            return False
        if params.get("drop_params") is True:
            return False
        return _is_unsupported_params_error(exc)

    def _call_with_retries(self, func, *, request_count: int) -> Any:
        retry_policy = self.config.retry_policy
        # max_retries counts retries after the initial attempt.
        attempts = max(1, retry_policy.max_retries + 1)

        for attempt in range(attempts):
            self._respect_rate_limit(request_count)
            try:
                return func()
            except Exception as exc:
                if attempt >= attempts - 1 or not _is_retryable_error(exc):
                    raise
                delay = min(
                    retry_policy.max_delay,
                    retry_policy.base_delay * (2 ** attempt),
                )
                if retry_policy.jitter > 0:
                    delay += random.uniform(0, delay * retry_policy.jitter)
                logger.warning(
                    "Retryable LLM error | Provider: {} | Model: {} | "
                    "Attempt: {}/{} | Waiting: {:.2f}s | Error: {}",
                    self.provider_id,
                    self.model_id,
                    attempt + 1,
                    attempts,
                    delay,
                    exc,
                )
                self._sleep(delay)

        raise RuntimeError("unreachable retry state")

    def _respect_rate_limit(self, request_count: int = 1) -> None:
        if self.config.rpm_limit is None:
            return

        # A request block larger than the whole budget can only run once the
        # window is empty; it is still recorded in full so later requests wait.
        needed = min(request_count, self.config.rpm_limit)

        while True:
            with self._rate_lock:
                now = time.monotonic()
                self._request_timestamps = [
                    timestamp
                    for timestamp in self._request_timestamps
                    if now - timestamp < 60
                ]
                if len(self._request_timestamps) + needed <= self.config.rpm_limit:
                    # Reserve before dispatch so concurrent workers and failed
                    # attempts count against the budget.
                    self._request_timestamps.extend([now] * request_count)
                    return
                earliest = self._request_timestamps[0]
                # +1s margin so we never release exactly on the boundary,
                # where the provider's server-side window may still count
                # the expiring request.
                sleep_time = max(0.0, 60 - (now - earliest)) + 1.0

            if sleep_time > 0:
                logger.warning(
                    "Rate limit reached | Provider: {} | Model: {} | "
                    "Waiting {:.2f}s",
                    self.provider_id,
                    self.model_id,
                    sleep_time,
                )
                self._sleep(sleep_time)

    def _parse_response(
        self,
        response: NormalizedResponse,
        *,
        response_format: type[T] | None,
    ) -> str | T:
        if response_format is None:
            return response.text.strip() if response.text else response.text

        parsed = getattr(response.raw, "output_parsed", None)
        if parsed is not None:
            return parsed

        content = self._strip_code_fences(response.text)
        try:
            return response_format.model_validate_json(content)
        except Exception as validation_error:
            content_preview = (
                content[:200] + "..." if len(content) > 200 else content
            )
            raise ValueError(
                f"Failed to parse JSON response into {response_format.__name__}.\n"
                f"Validation error: {validation_error}\n"
                f"Content received (first 200 chars):\n{content_preview}"
            ) from validation_error

    def _build_request_metadata(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_trace_metadata(
            model=self,
            component="served_model.generate",
            trace_name=f"datafast.{self.provider_id}",
            metadata=metadata,
        )

    def _get_model_string(self) -> str:
        prefix = f"{self.config.litellm_route}/"
        if self.model_id.startswith(prefix):
            return self.model_id
        return f"{prefix}{self.model_id}"

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        if not content:
            return content

        content = content.strip()
        if content.startswith("```"):
            first_newline = content.find("\n")
            content = content[first_newline + 1 :] if first_newline != -1 else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()


class _OpenAIServedModel(ServedModel):
    def __init__(self, model_id: str = "gpt-5.5", **kwargs: Any) -> None:
        super().__init__(
            "openai",
            model_id,
            litellm_route="openai",
            env_key_name="OPENAI_API_KEY",
            **kwargs,
        )


class _AnthropicServedModel(ServedModel):
    def __init__(self, model_id: str = "claude-haiku-4-5", **kwargs: Any) -> None:
        super().__init__(
            "anthropic",
            model_id,
            litellm_route="anthropic",
            env_key_name="ANTHROPIC_API_KEY",
            **kwargs,
        )


class _GeminiServedModel(ServedModel):
    def __init__(self, model_id: str = "gemini-3.5-flash-lite", **kwargs: Any) -> None:
        super().__init__(
            "gemini",
            model_id,
            litellm_route="gemini",
            env_key_name="GEMINI_API_KEY",
            **kwargs,
        )


class _MistralServedModel(ServedModel):
    # LiteLLM has no Files support for Mistral, so these two call the API directly.
    FILES_URL = "https://api.mistral.ai/v1/files"

    def __init__(self, model_id: str = "mistral-small-2603", **kwargs: Any) -> None:
        super().__init__(
            "mistral",
            model_id,
            litellm_route="mistral",
            env_key_name="MISTRAL_API_KEY",
            **kwargs,
        )

    def upload_file(
        self,
        path: str | Path,
        *,
        purpose: str = "ocr",
        expiry: int | None = None,
    ) -> str:
        """Upload a document to Mistral's Files API and return its id.

        Mistral's chat API accepts a document only as an id, so this is how one
        reaches the model:

            file_id = model.upload_file("report.pdf")
            ContentPart(type="file", url=file_id)

        Uploading is deliberately separate from generating: the id can be reused
        across every request in a pipeline run, and the upload stays visible rather
        than happening behind a `generate()` call. The file then remains in the
        Mistral account until `delete_file` removes it — Datafast does not track it.

        `expiry` asks Mistral to expire the file on its own, which is worth setting
        for throwaway uploads that no later run needs. It is forwarded verbatim:
        Mistral documents the field as an integer but not its unit, so the value
        means whatever the API says it means. Omitted by default, leaving the
        account's own retention in force.
        """
        path = Path(path)
        payload: dict[str, Any] = {"purpose": purpose}
        if expiry is not None:
            payload["expiry"] = expiry
        with path.open("rb") as handle:
            response = httpx.post(
                self.FILES_URL,
                headers=self._files_headers(),
                data=payload,
                files={"file": (path.name, handle)},
                timeout=self.timeout or 60.0,
            )
        response.raise_for_status()
        return response.json()["id"]

    def delete_file(self, file_id: str) -> None:
        """Delete a file previously returned by `upload_file`."""
        response = httpx.delete(
            f"{self.FILES_URL}/{file_id}",
            headers=self._files_headers(),
            timeout=self.timeout or 60.0,
        )
        response.raise_for_status()

    def _files_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "A Mistral API key is required to use the Files API. Set "
                "MISTRAL_API_KEY or pass api_key=."
            )
        return {"Authorization": f"Bearer {self.api_key}"}


class _OpenRouterServedModel(ServedModel):
    def __init__(self, model_id: str = "openai/gpt-5.4-mini", **kwargs: Any) -> None:
        super().__init__(
            "openrouter",
            model_id,
            litellm_route="openrouter",
            env_key_name="OPENROUTER_API_KEY",
            **kwargs,
        )


class _OllamaServedModel(ServedModel):
    # LiteLLM has no Ollama introspection, so the probe below calls the daemon.
    SHOW_PATH = "/api/show"
    DEFAULT_API_BASE = "http://localhost:11434"

    def __init__(self, model_id: str = "gemma4:12b", **kwargs: Any) -> None:
        super().__init__(
            "ollama",
            model_id,
            litellm_route="ollama_chat",
            env_key_name=None,
            **kwargs,
        )

    def probe_capabilities(self) -> frozenset[str]:
        """Ask the Ollama daemon what this model can actually do.

        Returns Ollama's own capability names — "completion", "vision", "audio",
        "thinking", "tools", "embedding", "insert" — not Datafast's vocabulary.

        Which model is pulled is a property of the machine, not of the id, so
        Datafast resolves an Ollama model from name heuristics and can only be
        approximately right: `OLLAMA_CHAT` declares `Modality.IMAGE` for every
        model, and a reasoning model whose name carries no marker resolves to the
        profile with reasoning switched off. The daemon knows the answer exactly
        and answers for free, so it is worth asking before assuming:

            if "vision" not in model.probe_capabilities():
                ...  # don't bother attaching the image

        Raises `httpx.HTTPStatusError` if the model is not pulled, and a connect
        error if no daemon is listening.
        """
        response = httpx.post(
            f"{self._resolved_api_base()}{self.SHOW_PATH}",
            json={"model": self.model_id},
            timeout=self.timeout or 60.0,
        )
        response.raise_for_status()
        return frozenset(response.json().get("capabilities") or ())

    def _resolved_api_base(self) -> str:
        """The daemon the generate calls reach, resolved the way LiteLLM resolves it
        (`llms/ollama/common_utils.py:76`) so the probe cannot end up on another host."""
        base = (
            self.api_base_url
            or os.getenv("OLLAMA_API_BASE")
            or self.DEFAULT_API_BASE
        )
        return base.rstrip("/")


class _OpenAICompatibleServedModel(ServedModel):
    def __init__(
        self,
        model_id: str,
        *,
        provider_id: str,
        litellm_route: str = "openai",
        env_key_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider_id,
            model_id,
            litellm_route=litellm_route,
            env_key_name=env_key_name,
            **kwargs,
        )


def openai(model_id: str = "gpt-5.5", **kwargs: Any) -> ServedModel:
    """Build a served model on OpenAI.

    Reads `OPENAI_API_KEY` unless `api_key=` is passed. Reasoning models are
    reached over the Responses endpoint and plain chat models over Chat
    Completions; `endpoint_mode=` overrides the choice.

    Args:
        model_id: OpenAI model name.
        **kwargs: Any `ServedModelConfig` field — `temperature`, `max_tokens`,
            `reasoning_effort`, `max_concurrent`, `rpm_limit`, `timeout`, ...

    Returns:
        A `ServedModel` bound to OpenAI.
    """
    return _OpenAIServedModel(model_id=model_id, **kwargs)


def anthropic(model_id: str = "claude-haiku-4-5", **kwargs: Any) -> ServedModel:
    """Build a served model on Anthropic.

    Reads `ANTHROPIC_API_KEY` unless `api_key=` is passed. Requests go through
    LiteLLM's own HTTP transport, so the `anthropic` SDK is not required.

    Args:
        model_id: Anthropic model name.
        **kwargs: Any `ServedModelConfig` field — `thinking`, `temperature`,
            `max_tokens`, `max_concurrent`, ...

    Returns:
        A `ServedModel` bound to Anthropic.
    """
    return _AnthropicServedModel(model_id=model_id, **kwargs)


def gemini(model_id: str = "gemini-3.5-flash-lite", **kwargs: Any) -> ServedModel:
    """Build a served model on Google Gemini.

    Reads `GEMINI_API_KEY` unless `api_key=` is passed. Requests go through
    LiteLLM's own HTTP transport, so `google-generativeai` is not required.

    Args:
        model_id: Gemini model name.
        **kwargs: Any `ServedModelConfig` field — `thinking`, `temperature`,
            `max_tokens`, `max_concurrent`, ...

    Returns:
        A `ServedModel` bound to Gemini.
    """
    return _GeminiServedModel(model_id=model_id, **kwargs)


def mistral(model_id: str = "mistral-small-2603", **kwargs: Any) -> ServedModel:
    """Build a served model on Mistral.

    Reads `MISTRAL_API_KEY` unless `api_key=` is passed.

    Args:
        model_id: Mistral model name. `magistral` and `-reasoning` models
            resolve to a reasoning-capable profile.
        **kwargs: Any `ServedModelConfig` field — `temperature`, `max_tokens`,
            `max_concurrent`, ...

    Returns:
        A `ServedModel` bound to Mistral.
    """
    return _MistralServedModel(model_id=model_id, **kwargs)


def openrouter(
    model_id: str = "openai/gpt-5.4-mini",
    **kwargs: Any,
) -> ServedModel:
    """Build a served model on OpenRouter.

    Reads `OPENROUTER_API_KEY` unless `api_key=` is passed. OpenRouter fronts
    many upstream providers, so `model_id` carries a `vendor/model` prefix.

    Args:
        model_id: OpenRouter model name, e.g. `"openai/gpt-5.4-mini"`.
        **kwargs: Any `ServedModelConfig` field — `temperature`, `max_tokens`,
            `max_concurrent`, ...

    Returns:
        A `ServedModel` bound to OpenRouter.
    """
    return _OpenRouterServedModel(model_id=model_id, **kwargs)


def ollama(model_id: str = "gemma4:12b", **kwargs: Any) -> ServedModel:
    """Build a served model on a local Ollama daemon.

    Needs no API key. The daemon is reached at `http://localhost:11434` unless
    `OLLAMA_API_BASE` or `api_base_url=` says otherwise; capabilities are read
    from the daemon itself rather than a static catalogue.

    Args:
        model_id: Ollama model tag, e.g. `"gemma4:12b"`.
        **kwargs: Any `ServedModelConfig` field. Use `repeat_penalty` rather
            than `frequency_penalty` — Ollama's knob is a multiplier neutral
            at `1.0`.

    Returns:
        A `ServedModel` bound to the local Ollama daemon.
    """
    return _OllamaServedModel(model_id=model_id, **kwargs)


def openai_compatible(
    model_id: str,
    *,
    provider_id: str,
    api_base_url: str | None = None,
    **kwargs: Any,
) -> ServedModel:
    """Build a served model reached over the OpenAI-compatible transport.

    provider_id names the server doing the serving ('vllm', 'llamacpp', ...),
    never the wire format. Servers without a capability profile of their own
    still work; they resolve to the conservative OpenAI-compatible profile.
    """
    return _OpenAICompatibleServedModel(
        model_id=model_id,
        provider_id=_normalize_provider_id(provider_id),
        api_base_url=api_base_url,
        **kwargs,
    )


# An OpenAI-shaped wire format says nothing about which server is on the other
# end, so these never identify a provider.
_WIRE_FORMAT_IDS = frozenset({"openai_compatible", "openai_api", "oai_compatible"})

_PROVIDER_ID_ALIASES = {
    "llama_cpp": "llamacpp",
    "llama.cpp": "llamacpp",
}


def _normalize_provider_id(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    normalized = _PROVIDER_ID_ALIASES.get(normalized, normalized)
    if not normalized:
        raise ValueError(
            "provider_id must name the server that serves the model, e.g. 'vllm'."
        )
    if normalized in _WIRE_FORMAT_IDS:
        raise ValueError(
            f"provider_id '{value}' names a wire format, not a server. Pass the "
            "server that serves the model, e.g. provider_id='vllm' or "
            "provider_id='llamacpp'."
        )
    return normalized


def _coerce_endpoint_mode(value: str | EndpointMode) -> EndpointMode:
    if isinstance(value, EndpointMode):
        return value
    try:
        return EndpointMode(value)
    except ValueError as exc:
        raise ValueError("endpoint_mode must be 'auto', 'chat', or 'responses'") from exc


def _coerce_unsupported_policy(
    value: str | UnsupportedParamsPolicy,
) -> UnsupportedParamsPolicy:
    if isinstance(value, UnsupportedParamsPolicy):
        return value
    try:
        return UnsupportedParamsPolicy(value)
    except ValueError as exc:
        raise ValueError("unsupported_params must be 'fail', 'warn', or 'quiet'") from exc


def _normalize_metadata(
    metadata: list[dict[str, Any] | None] | dict[str, Any] | None,
    expected_length: int,
) -> list[dict[str, Any] | None]:
    if isinstance(metadata, list):
        if len(metadata) != expected_length:
            raise ValueError("metadata length must match messages length")
        return metadata
    return [metadata] * expected_length


def _combine_batch_metadata(requests: list[NormalizedRequest]) -> dict[str, Any]:
    metadata_items = [request.metadata for request in requests]
    return {
        "datafast_batch_size": len(requests),
        "datafast_batch_metadata": metadata_items,
    }


def _is_single_messages(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], dict)


def _is_batch_messages(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list)


def _normalize_message(message: Message, *, include_media_uuid: bool = False) -> Message:
    if not isinstance(message, dict):
        raise ValueError("Each message must be a dictionary")

    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [
            _normalize_content_part(part, include_media_uuid=include_media_uuid)
            for part in content
        ]
    elif content is not None and not isinstance(content, str):
        raise ValueError("message content must be a string, list of parts, or None")
    return normalized


def _normalize_content_part(
    part: Any,
    *,
    include_media_uuid: bool = False,
) -> dict[str, Any]:
    part = _content_part_to_dict(part)
    part_type = part.get("type")

    if part_type in {"image_url", "input_audio", "video_url"}:
        return _without_none(part)
    if part_type == "text":
        return _normalize_text_part(part)
    if part_type == "image":
        return _normalize_image_part(part, include_media_uuid=include_media_uuid)
    if part_type == "audio":
        return _normalize_audio_part(part)
    if part_type == "video":
        return _normalize_video_part(part, include_media_uuid=include_media_uuid)
    if part_type in {"file", "document"}:
        return _normalize_file_part(part)
    return part


def _content_part_to_dict(part: Any) -> dict[str, Any]:
    if isinstance(part, ContentPart):
        part = {
            "type": part.type,
            "text": part.text,
            "url": part.url,
            "data": part.data,
            "media_type": part.media_type,
            "media_id": part.media_id,
            "filename": part.filename,
            **part.provider_options,
        }

    if not isinstance(part, dict):
        raise ValueError("content parts must be dictionaries or ContentPart objects")
    return part


def _normalize_text_part(part: dict[str, Any]) -> dict[str, Any]:
    return _without_none({"type": "text", "text": part.get("text")})


def _normalize_image_part(
    part: dict[str, Any],
    *,
    include_media_uuid: bool = False,
) -> dict[str, Any]:
    image_url: dict[str, Any] = {
        "url": _media_url_from_part(part, kind="image"),
    }
    if part.get("format") or part.get("media_type"):
        image_url["format"] = part.get("format") or part.get("media_type")
    if part.get("detail"):
        image_url["detail"] = part["detail"]
    normalized = {"type": "image_url", "image_url": _without_none(image_url)}
    if include_media_uuid and part.get("media_id"):
        normalized["uuid"] = part["media_id"]
    return normalized


def _normalize_audio_part(part: dict[str, Any]) -> dict[str, Any]:
    data = part.get("data")
    if not data:
        raise ValueError(
            "audio content parts require base64 'data'; URL-only audio input "
            "is not supported by chat audio APIs"
        )
    audio_format = part.get("format") or part.get("media_type") or "wav"
    # Accept MIME types ("audio/wav") but send the bare format ("wav").
    if "/" in audio_format:
        audio_format = audio_format.split("/", 1)[1]
    return {
        "type": "input_audio",
        "input_audio": {"data": data, "format": audio_format},
    }


def _normalize_video_part(
    part: dict[str, Any],
    *,
    include_media_uuid: bool = False,
) -> dict[str, Any]:
    normalized = {
        "type": "video_url",
        "video_url": {"url": _media_url_from_part(part, kind="video")},
    }
    if include_media_uuid and part.get("media_id"):
        normalized["uuid"] = part["media_id"]
    return normalized


def _media_url_from_part(part: dict[str, Any], *, kind: str) -> str:
    url = part.get("url")
    if url:
        return url

    if part.get("data"):
        return _data_uri_from_part(part, kind=kind)

    raise ValueError(f"{kind} content parts require either 'url' or 'data'")


def _data_uri_from_part(part: dict[str, Any], *, kind: str) -> str:
    """Wrap raw base64 in a `data:` URI; pass an existing one through."""
    data = part["data"]
    if data.startswith("data:"):
        return data
    media_type = part.get("media_type") or part.get("format")
    if not media_type:
        raise ValueError(
            f"{kind} content parts with raw base64 'data' need 'media_type' "
            "(e.g. 'image/png', 'application/pdf') to build a data URI, or "
            "pass a full 'data:' URI directly"
        )
    return f"data:{media_type};base64,{data}"


def _normalize_file_part(part: dict[str, Any]) -> dict[str, Any]:
    if isinstance(part.get("file"), dict):
        file_payload = part["file"]
    elif part.get("data"):
        # OpenAI's Responses API rejects inline file data without a filename.
        file_payload = {
            "file_data": _data_uri_from_part(part, kind="file"),
            "filename": part.get("filename"),
        }
    else:
        file_payload = {"file_id": part.get("url")}
    return {"type": "file", "file": _without_none(file_payload)}


def _to_responses_input(messages: Messages) -> Messages:
    """Convert chat-normalized messages into Responses API input items.

    The Responses API rejects chat-completions part types: it expects
    input_text/input_image/input_file for user input and output_text for
    assistant history. String content passes through unchanged.
    """
    converted: Messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            converted.append(message)
            continue

        text_type = (
            "output_text" if message.get("role") == "assistant" else "input_text"
        )
        parts = [_to_responses_part(part, text_type=text_type) for part in content]
        new_message = dict(message)
        new_message["content"] = parts
        converted.append(new_message)
    return converted


def _to_responses_part(part: dict[str, Any], *, text_type: str) -> dict[str, Any]:
    part_type = part.get("type")

    if part_type == "text":
        return {"type": text_type, "text": part.get("text")}

    if part_type == "image_url":
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        new_part: dict[str, Any] = {"type": "input_image", "image_url": url}
        if isinstance(image_url, dict) and image_url.get("detail"):
            new_part["detail"] = image_url["detail"]
        if part.get("uuid"):
            new_part["uuid"] = part["uuid"]
        return new_part

    if part_type == "file":
        payload = dict(part.get("file") or {})
        new_part = {"type": "input_file"}
        file_id = payload.pop("file_id", None)
        if isinstance(file_id, str) and file_id.startswith(("http://", "https://")):
            new_part["file_url"] = file_id
        elif file_id is not None:
            new_part["file_id"] = file_id
        new_part.update(payload)
        return new_part

    return part


def _modality_for_part(part: dict[str, Any]) -> Modality:
    part_type = part.get("type")
    if part_type == "text":
        return Modality.TEXT
    if part_type in {"image", "image_url"}:
        return Modality.IMAGE
    if part_type in {"audio", "input_audio"}:
        return Modality.AUDIO
    if part_type in {"video", "video_url"}:
        return Modality.VIDEO
    # 'document' parts normalize to 'file' before gating, and no served model
    # declares Modality.DOCUMENT, so both gate as FILE.
    if part_type in {"file", "document"}:
        return Modality.FILE
    return Modality.TEXT


def _append_json_instructions(messages: Messages) -> None:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + JSON_INSTRUCTIONS
            return
        if isinstance(content, list):
            for part in reversed(content):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = part["text"] + JSON_INSTRUCTIONS
                    return
    messages.append({"role": "user", "content": JSON_INSTRUCTIONS.strip()})


def _extract_chat_text(response: Any) -> str:
    choice = _get_first_choice(response)
    if choice is None:
        raise RuntimeError(
            f"Unexpected chat response from LiteLLM: {type(response).__name__}"
        )

    message = _get_attr_or_key(choice, "message")
    if message is None:
        text = _get_attr_or_key(choice, "text")
        return "" if text is None else str(text)

    content = _get_attr_or_key(message, "content")
    return _content_to_text(content)


def _extract_chat_reasoning_content(response: Any) -> str | None:
    message = _extract_chat_message(response)
    if message is None:
        return None

    reasoning_content = _get_attr_or_key(message, "reasoning_content")
    if reasoning_content is None:
        reasoning_content = _get_attr_or_key(message, "reasoning")
    if reasoning_content is None:
        return None
    if isinstance(reasoning_content, list):
        return _content_to_text(reasoning_content).strip() or None
    return str(reasoning_content).strip() or None


def _extract_chat_thinking_blocks(response: Any) -> list[dict[str, Any]]:
    message = _extract_chat_message(response)
    if message is None:
        return []

    blocks = _get_attr_or_key(message, "thinking_blocks")
    if not blocks:
        return []
    if not isinstance(blocks, list):
        blocks = [blocks]
    return [_normalize_mapping_block(block) for block in blocks]


def _extract_chat_images(response: Any) -> list[dict[str, Any]]:
    message = _extract_chat_message(response)
    if message is None:
        return []

    images = _get_attr_or_key(message, "images")
    collected = list(_normalize_optional_list(images))

    content = _get_attr_or_key(message, "content")
    for part in _normalize_optional_list(content):
        part_type = _get_attr_or_key(part, "type")
        if part_type in {"image", "image_url", "output_image"}:
            collected.append(part)

    return [_normalize_mapping_block(image) for image in collected]


def _extract_chat_audio(response: Any) -> dict[str, Any] | None:
    message = _extract_chat_message(response)
    if message is None:
        return None

    audio = _get_attr_or_key(message, "audio")
    if audio:
        return _normalize_mapping_block(audio)

    content = _get_attr_or_key(message, "content")
    for part in _normalize_optional_list(content):
        part_type = _get_attr_or_key(part, "type")
        if part_type in {"audio", "output_audio"}:
            return _normalize_mapping_block(part)
    return None


def _extract_chat_message(response: Any) -> Any:
    choice = _get_first_choice(response)
    if choice is None:
        return None
    return _get_attr_or_key(choice, "message")


def _extract_responses_text(response: Any) -> str:
    output_text = _get_attr_or_key(response, "output_text")
    if output_text:
        return str(output_text)

    output = _normalize_optional_list(_get_attr_or_key(response, "output"))
    texts: list[str] = []
    for item in output:
        content = _get_attr_or_key(item, "content") or []
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in _normalize_optional_list(content):
            part_type = _get_attr_or_key(part, "type")
            if part_type in {"output_text", "text"}:
                text = _get_attr_or_key(part, "text")
                if text is not None:
                    texts.append(str(text))
    if texts:
        return "".join(texts)
    if output:
        return ""
    raise RuntimeError(
        f"Unexpected Responses API response from LiteLLM: {type(response).__name__}"
    )


def _extract_responses_reasoning(response: Any) -> str | None:
    reasoning_content = _get_attr_or_key(response, "reasoning_content")
    if reasoning_content:
        return str(reasoning_content).strip() or None

    output = _normalize_optional_list(_get_attr_or_key(response, "output"))
    texts: list[str] = []
    for item in output:
        item_type = _get_attr_or_key(item, "type")
        if item_type != "reasoning":
            continue

        for field_name in ("text", "content"):
            value = _get_attr_or_key(item, field_name)
            if value:
                texts.append(_content_to_text(value))

        summary = _get_attr_or_key(item, "summary") or []
        if isinstance(summary, str):
            texts.append(summary)
            continue
        for part in _normalize_optional_list(summary):
            text = _get_attr_or_key(part, "text") or _get_attr_or_key(part, "content")
            if text:
                texts.append(_content_to_text(text))

    joined = "\n".join(text.strip() for text in texts if text and text.strip())
    return joined or None


def _extract_responses_output_items(response: Any) -> list[dict[str, Any]]:
    output = _get_attr_or_key(response, "output") or []
    return [_normalize_mapping_block(item) for item in _normalize_optional_list(output)]


def _extract_responses_images(response: Any) -> list[dict[str, Any]]:
    images: list[Any] = []
    for item in _normalize_optional_list(_get_attr_or_key(response, "output")):
        item_type = _get_attr_or_key(item, "type")
        if item_type in {"image", "output_image", "image_generation_call"}:
            images.append(item)
        for part in _normalize_optional_list(_get_attr_or_key(item, "content")):
            part_type = _get_attr_or_key(part, "type")
            if part_type in {"image", "image_url", "output_image"}:
                images.append(part)
    return [_normalize_mapping_block(image) for image in images]


def _extract_responses_audio(response: Any) -> dict[str, Any] | None:
    for item in _normalize_optional_list(_get_attr_or_key(response, "output")):
        item_type = _get_attr_or_key(item, "type")
        if item_type in {"audio", "output_audio"}:
            return _normalize_mapping_block(item)
        for part in _normalize_optional_list(_get_attr_or_key(item, "content")):
            part_type = _get_attr_or_key(part, "type")
            if part_type in {"audio", "output_audio"}:
                return _normalize_mapping_block(part)
    return None


def _get_first_choice(response: Any) -> Any:
    choices = _get_attr_or_key(response, "choices")
    if not choices:
        return None
    return choices[0]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            text = _get_attr_or_key(part, "text")
            if text is not None:
                texts.append(str(text))
        return "".join(texts)
    return str(content)


def _get_attr_or_key(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_optional_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_mapping_block(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped

    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dumped

    result: dict[str, Any] = {}
    for name in ("type", "text", "thinking", "content", "signature"):
        attr = getattr(value, name, None)
        if attr is not None:
            result[name] = attr
    if result:
        return result
    return {"content": str(value)}


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _is_retryable_error(exc: Exception) -> bool:
    retryable_types = (
        litellm_exceptions.RateLimitError,
        litellm_exceptions.APIConnectionError,
        litellm_exceptions.Timeout,
        litellm_exceptions.InternalServerError,
        litellm_exceptions.ServiceUnavailableError,
    )
    return isinstance(exc, retryable_types)


def _is_unsupported_params_error(exc: Exception) -> bool:
    unsupported_type = getattr(litellm_exceptions, "UnsupportedParamsError", None)
    if unsupported_type is not None and isinstance(exc, unsupported_type):
        return True
    return exc.__class__.__name__ == "UnsupportedParamsError"


__all__ = [
    "ServedModel",
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "openrouter",
    "ollama",
    "openai_compatible",
]
