"""Capability-aware LLM providers for Datafast."""

from __future__ import annotations

import copy
import os
import random
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, TypeVar

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
    TargetCapabilities,
    TargetConfig,
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


class LLMProvider:
    """One Datafast provider target resolved to LiteLLM request adapters."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        *,
        litellm_provider: str,
        env_key_name: str | None,
        endpoint_mode: str | EndpointMode = EndpointMode.AUTO,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
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
        capabilities: TargetCapabilities | None = None,
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

        self.config = TargetConfig(
            provider=provider,
            model_id=model_id,
            litellm_provider=litellm_provider,
            env_key_name=env_key_name,
            endpoint_mode=_coerce_endpoint_mode(endpoint_mode),
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
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
            provider,
            model_id,
            api_base_url=api_base_url,
            explicit=capabilities,
        )
        self.endpoint_mode = self._resolve_endpoint_mode(self.config.endpoint_mode)

        self.provider_name = provider
        self.model_id = model_id
        self.env_key_name = env_key_name
        self.api_key = api_key or (os.getenv(env_key_name) if env_key_name else None)
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
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
            self.provider_name,
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
        try:
            results = self._generate_requests(requests, response_format=response_format)
        except ValueError:
            raise
        except Exception as exc:
            error_trace = traceback.format_exc()
            logger.error(
                "Generation failed | Provider: {} | Model: {} | Error: {}",
                self.provider_name,
                self.model_id,
                exc,
            )
            raise RuntimeError(
                f"Error generating response with {self.provider_name}:\n{error_trace}"
            ) from exc

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
        previous_ids = previous_response_ids or [None] * len(messages)
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
        responses = self._generate_normalized_responses(
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
        previous_ids = previous_response_ids or [None] * len(messages)
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
        return self._generate_normalized_responses(requests, response_format=None)

    def _generate_requests(
        self,
        requests: list[NormalizedRequest],
        *,
        response_format: type[T] | None,
    ) -> list[str] | list[T]:
        responses = self._generate_normalized_responses(
            requests,
            response_format=response_format,
        )
        return [
            self._parse_response(response, response_format=response_format)
            for response in responses
        ]

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
                f"{self.provider_name}/{self.model_id} does not expose native "
                "same-target batching for this endpoint. Falling back to bounded "
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
        params = self._build_chat_params(
            NormalizedRequest(
                messages=[],
                metadata=_combine_batch_metadata(requests),
            ),
            response_format,
        )
        params["messages"] = [request.messages for request in requests]
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
                raise RuntimeError(f"Batch item {index} failed: {item}") from item
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
            self._add_supported_param(
                params,
                "previous_response_id",
                request.previous_response_id,
                endpoint=EndpointMode.CHAT,
            )
        self._add_transport_params(params, endpoint=EndpointMode.CHAT)
        self._add_common_generation_params(params, endpoint=EndpointMode.CHAT)
        self._add_chat_structured_output(params, response_format)
        params.update(self.config.provider_params)
        return _without_none(params)

    def _build_responses_params(
        self,
        request: NormalizedRequest,
        response_format: type[T] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._get_model_string(),
            "input": request.messages,
            "metadata": self._build_request_metadata(request.metadata),
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
        self._add_supported_param(
            params,
            "temperature",
            self.config.temperature,
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
            target_name=token_param,
        )

        if self.config.thinking is False:
            return

        effort = self.config.reasoning_effort
        if effort is None and self.config.thinking is True:
            effort = "low"

        if endpoint == EndpointMode.RESPONSES and effort is not None:
            self._add_supported_param(
                params,
                "reasoning_effort",
                {"effort": effort},
                endpoint=endpoint,
                target_name="reasoning",
            )
            return

        self._add_supported_param(
            params,
            "reasoning_effort",
            effort,
            endpoint=endpoint,
        )

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
            if self.provider_name == "ollama":
                params["format"] = "json"
        elif mode == StructuredOutputMode.PROMPTED_JSON:
            warnings.warn(
                (
                    f"{self.provider_name}/{self.model_id} has no declared native "
                    "schema support. Using prompted JSON plus Pydantic validation."
                ),
                UserWarning,
                stacklevel=3,
            )
        else:
            raise ValueError(
                f"{self.provider_name}/{self.model_id} does not support structured output"
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
                f"{self.provider_name}/{self.model_id} does not support native "
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
        if self.api_key is not None:
            params["api_key"] = self.api_key
        elif self.env_key_name and not self.capabilities.no_api_key:
            env_key = os.getenv(self.env_key_name)
            if env_key:
                params["api_key"] = env_key
            else:
                raise ValueError(
                    f"{self.env_key_name} environment variable not set. "
                    "Set it or provide api_key when initializing the provider."
                )

    def _add_supported_param(
        self,
        params: dict[str, Any],
        source_name: str,
        value: Any,
        *,
        endpoint: EndpointMode,
        target_name: str | None = None,
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

        params[target_name or source_name] = value

    def _handle_unsupported_param(self, name: str) -> None:
        message = (
            f"Parameter '{name}' is not supported by resolved target "
            f"{self.provider_name}/{self.model_id} and will be omitted."
        )
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

        normalized = [_normalize_message(message) for message in copy.deepcopy(messages)]
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
                        f"{self.provider_name}/{self.model_id}"
                    )

    def _resolve_endpoint_mode(self, endpoint_mode: EndpointMode) -> EndpointMode:
        if endpoint_mode == EndpointMode.AUTO:
            return self.capabilities.default_endpoint_mode
        if not self.capabilities.supports_endpoint(endpoint_mode):
            raise ValueError(
                f"{self.provider_name}/{self.model_id} does not support "
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
        attempts = max(1, retry_policy.max_retries)

        for attempt in range(attempts):
            self._respect_rate_limit(request_count)
            try:
                response = func()
                self._record_requests(request_count)
                return response
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
                    self.provider_name,
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

        with self._rate_lock:
            now = time.monotonic()
            self._request_timestamps = [
                timestamp
                for timestamp in self._request_timestamps
                if now - timestamp < 60
            ]

            while len(self._request_timestamps) + request_count > self.config.rpm_limit:
                earliest = self._request_timestamps[0]
                sleep_time = max(0.0, 60 - (now - earliest))
                if sleep_time > 0:
                    logger.warning(
                        "Rate limit reached | Provider: {} | Model: {} | "
                        "Waiting {:.2f}s",
                        self.provider_name,
                        self.model_id,
                        sleep_time,
                    )
                    self._sleep(sleep_time)
                now = time.monotonic()
                self._request_timestamps = [
                    timestamp
                    for timestamp in self._request_timestamps
                    if now - timestamp < 60
                ]

    def _record_requests(self, request_count: int = 1) -> None:
        if self.config.rpm_limit is None:
            return
        with self._rate_lock:
            now = time.monotonic()
            self._request_timestamps.extend([now] * request_count)

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
            component="provider.generate",
            trace_name=f"datafast.{self.provider_name}",
            metadata=metadata,
        )

    def _get_model_string(self) -> str:
        prefix = f"{self.config.litellm_provider}/"
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


class OpenAIProvider(LLMProvider):
    def __init__(self, model_id: str = "gpt-5.5", **kwargs: Any) -> None:
        super().__init__(
            "openai",
            model_id,
            litellm_provider="openai",
            env_key_name="OPENAI_API_KEY",
            **kwargs,
        )


class AnthropicProvider(LLMProvider):
    def __init__(self, model_id: str = "claude-haiku-4-5", **kwargs: Any) -> None:
        super().__init__(
            "anthropic",
            model_id,
            litellm_provider="anthropic",
            env_key_name="ANTHROPIC_API_KEY",
            **kwargs,
        )


class GeminiProvider(LLMProvider):
    def __init__(self, model_id: str = "gemini-3.1-flash-lite", **kwargs: Any) -> None:
        super().__init__(
            "gemini",
            model_id,
            litellm_provider="gemini",
            env_key_name="GEMINI_API_KEY",
            **kwargs,
        )


class MistralProvider(LLMProvider):
    def __init__(self, model_id: str = "mistral-small-2603", **kwargs: Any) -> None:
        super().__init__(
            "mistral",
            model_id,
            litellm_provider="mistral",
            env_key_name="MISTRAL_API_KEY",
            **kwargs,
        )


class OpenRouterProvider(LLMProvider):
    def __init__(self, model_id: str = "openai/gpt-5.4-mini", **kwargs: Any) -> None:
        super().__init__(
            "openrouter",
            model_id,
            litellm_provider="openrouter",
            env_key_name="OPENROUTER_API_KEY",
            **kwargs,
        )


class OllamaProvider(LLMProvider):
    def __init__(self, model_id: str = "gemma3:4b", **kwargs: Any) -> None:
        super().__init__(
            "ollama",
            model_id,
            litellm_provider="ollama_chat",
            env_key_name=None,
            **kwargs,
        )


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model_id: str,
        *,
        provider: str = "openai_compatible",
        litellm_provider: str = "openai",
        env_key_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            provider,
            model_id,
            litellm_provider=litellm_provider,
            env_key_name=env_key_name,
            **kwargs,
        )


def openai(model_id: str = "gpt-5.5", **kwargs: Any) -> OpenAIProvider:
    return OpenAIProvider(model_id=model_id, **kwargs)


def anthropic(model_id: str = "claude-haiku-4-5", **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(model_id=model_id, **kwargs)


def gemini(model_id: str = "gemini-3.1-flash-lite", **kwargs: Any) -> GeminiProvider:
    return GeminiProvider(model_id=model_id, **kwargs)


def mistral(model_id: str = "mistral-small-2603", **kwargs: Any) -> MistralProvider:
    return MistralProvider(model_id=model_id, **kwargs)


def openrouter(
    model_id: str = "openai/gpt-5.4-mini",
    **kwargs: Any,
) -> OpenRouterProvider:
    return OpenRouterProvider(model_id=model_id, **kwargs)


def ollama(model_id: str = "gemma3:4b", **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(model_id=model_id, **kwargs)


def openai_compatible(
    model_id: str,
    *,
    api_base_url: str | None = None,
    backend: str = "openai_compatible",
    **kwargs: Any,
) -> OpenAICompatibleProvider:
    provider = _normalize_openai_compatible_backend(backend)
    return OpenAICompatibleProvider(
        model_id=model_id,
        provider=provider,
        api_base_url=api_base_url,
        **kwargs,
    )


def _normalize_openai_compatible_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "openai-compatible": "openai_compatible",
        "openai_compatible": "openai_compatible",
        "llama.cpp": "llamacpp",
        "llama_cpp": "llamacpp",
        "llamacpp": "llamacpp",
        "vllm": "vllm",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unsupported OpenAI-compatible backend '{value}'. Choose: {valid}"
        ) from exc


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


def _normalize_message(message: Message) -> Message:
    if not isinstance(message, dict):
        raise ValueError("Each message must be a dictionary")

    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [_normalize_content_part(part) for part in content]
    elif content is not None and not isinstance(content, str):
        raise ValueError("message content must be a string, list of parts, or None")
    return normalized


def _normalize_content_part(part: Any) -> dict[str, Any]:
    part = _content_part_to_dict(part)
    part_type = part.get("type")

    normalizers = {
        "text": _normalize_text_part,
        "image": _normalize_image_part,
        "audio": _normalize_audio_part,
        "video": _normalize_video_part,
        "file": _normalize_file_part,
        "document": _normalize_file_part,
    }
    if part_type in {"image_url", "input_audio", "video_url"}:
        return _without_none(part)
    if part_type in normalizers:
        return normalizers[part_type](part)
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
            **part.provider_options,
        }

    if not isinstance(part, dict):
        raise ValueError("content parts must be dictionaries or ContentPart objects")
    return part


def _normalize_text_part(part: dict[str, Any]) -> dict[str, Any]:
    return _without_none({"type": "text", "text": part.get("text")})


def _normalize_image_part(part: dict[str, Any]) -> dict[str, Any]:
    image_url: dict[str, Any] = {"url": part.get("url") or part.get("data")}
    if part.get("format") or part.get("media_type"):
        image_url["format"] = part.get("format") or part.get("media_type")
    if part.get("detail"):
        image_url["detail"] = part["detail"]
    normalized = {"type": "image_url", "image_url": _without_none(image_url)}
    if part.get("media_id"):
        normalized["uuid"] = part["media_id"]
    return normalized


def _normalize_audio_part(part: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "input_audio",
        "input_audio": _without_none({
            "data": part.get("data"),
            "format": part.get("format") or part.get("media_type") or "wav",
        }),
    }


def _normalize_video_part(part: dict[str, Any]) -> dict[str, Any]:
    normalized = {"type": "video_url", "video_url": {"url": part.get("url")}}
    if part.get("media_id"):
        normalized["uuid"] = part["media_id"]
    return normalized


def _normalize_file_part(part: dict[str, Any]) -> dict[str, Any]:
    if isinstance(part.get("file"), dict):
        file_payload = part["file"]
    elif part.get("data"):
        file_payload = {"file_data": part.get("data")}
    else:
        file_payload = {"file_id": part.get("url")}
    return {"type": "file", "file": _without_none(file_payload)}


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
    if part_type == "file":
        return Modality.FILE
    if part_type == "document":
        return Modality.DOCUMENT
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
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MistralProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "openrouter",
    "ollama",
    "openai_compatible",
]
