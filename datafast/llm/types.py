"""Shared types for Datafast served models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Message = dict[str, Any]
Messages = list[Message]

ContentPartType = Literal["text", "image", "audio", "video", "file", "document"]


class EndpointMode(str, Enum):
    AUTO = "auto"
    CHAT = "chat"
    RESPONSES = "responses"


class UnsupportedParamsPolicy(str, Enum):
    FAIL = "fail"
    WARN = "warn"
    QUIET = "quiet"


class StructuredOutputMode(str, Enum):
    NONE = "none"
    PROMPTED_JSON = "prompted_json"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class BatchMode(str, Enum):
    NONE = "none"
    LITELLM_BATCH = "litellm_batch"
    FALLBACK_CONCURRENCY = "fallback_concurrency"


class CacheMode(str, Enum):
    NONE = "none"
    PROVIDER_PROMPT = "provider_prompt"
    ROUTER = "router"
    LOCAL_KV = "local_kv"
    CLIENT_RESULT = "client_result"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    DOCUMENT = "document"


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25


@dataclass(frozen=True)
class ServedModelCapabilities:
    endpoint_modes: frozenset[EndpointMode]
    default_endpoint_mode: EndpointMode
    supported_params: frozenset[str] = frozenset()
    modalities: frozenset[Modality] = frozenset({Modality.TEXT})
    structured_output: StructuredOutputMode = StructuredOutputMode.PROMPTED_JSON
    batch_mode: BatchMode = BatchMode.FALLBACK_CONCURRENCY
    cache_mode: CacheMode = CacheMode.NONE
    supports_reasoning: bool = False
    supports_thinking: bool = False
    reasoning_requires_allowlist: bool = False
    # reasoning_effort value that thinking=True maps to.
    reasoning_effort_on: str = "low"
    # (name, value) request param that turns reasoning off for thinking=False.
    # None omits it, which leaves the served model's own default in force —
    # correct only where that default is "no reasoning".
    reasoning_off_param: tuple[str, Any] | None = None
    # Accepted reasoning_effort values, or None to forward any value unchecked.
    reasoning_efforts: frozenset[str] | None = None
    # True where the served model rejects a caller-chosen temperature once
    # reasoning is on, so temperature is dropped for those requests.
    reasoning_locks_temperature: bool = False
    supports_media_uuid: bool = False
    no_api_key: bool = False
    requires_chat_template: bool = False
    notes: tuple[str, ...] = ()

    def supports_endpoint(self, endpoint_mode: EndpointMode) -> bool:
        return endpoint_mode in self.endpoint_modes


@dataclass(frozen=True)
class ServedModelConfig:
    provider_id: str
    model_id: str
    litellm_route: str
    env_key_name: str | None
    endpoint_mode: EndpointMode = EndpointMode.AUTO
    temperature: float | None = None
    max_completion_tokens: int | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None
    rpm_limit: int | None = None
    timeout: float | None = None
    api_key: str | None = None
    api_base_url: str | None = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    unsupported_params: UnsupportedParamsPolicy = UnsupportedParamsPolicy.WARN
    provider_params: dict[str, Any] = field(default_factory=dict)
    max_concurrent: int = 4


@dataclass(frozen=True)
class NormalizedRequest:
    messages: Messages
    metadata: dict[str, Any] | None = None
    previous_response_id: str | None = None


@dataclass(frozen=True)
class NormalizedResponse:
    text: str
    raw: Any
    reasoning_content: str | None = None
    thinking_blocks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    audio: dict[str, Any] | None = None
    output_items: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ContentPart:
    type: ContentPartType
    text: str | None = None
    url: str | None = None
    data: str | None = None
    media_type: str | None = None
    media_id: str | None = None
    # Required by OpenAI's Responses API alongside inline file data.
    filename: str | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BatchMode",
    "CacheMode",
    "ContentPart",
    "ContentPartType",
    "EndpointMode",
    "Message",
    "Messages",
    "Modality",
    "NormalizedRequest",
    "NormalizedResponse",
    "RetryPolicy",
    "ServedModelCapabilities",
    "ServedModelConfig",
    "StructuredOutputMode",
    "UnsupportedParamsPolicy",
]
