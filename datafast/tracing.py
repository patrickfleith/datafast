"""Optional Langfuse tracing helpers for Datafast."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import warnings
from typing import Any

import litellm
from dotenv import load_dotenv

try:
    _DATAFAST_VERSION = importlib.metadata.version("datafast")
except importlib.metadata.PackageNotFoundError:
    _DATAFAST_VERSION = "0.0.0"

_ENV_LOADED = False
_LANGFUSE_CALLBACK = "langfuse"
_LANGFUSE_AUTO_DISABLED = False
_MISSING_LANGFUSE_WARNING_EMITTED = False


def load_env_once() -> None:
    """Load environment variables from `.env` exactly once."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    load_dotenv(override=False)
    _ENV_LOADED = True


def is_langfuse_tracing_enabled() -> bool:
    """Return whether Langfuse callbacks are active in LiteLLM."""
    return (
        _LANGFUSE_CALLBACK in litellm.success_callback
        and _LANGFUSE_CALLBACK in litellm.failure_callback
    )


def configure_langfuse_tracing(
    *,
    enabled: bool = True,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    release: str | None = None,
    debug: bool | str | None = None,
    load_env: bool = True,
    strict: bool = True,
) -> bool:
    """Enable or disable Langfuse tracing for LiteLLM-backed Datafast calls."""
    if load_env:
        load_env_once()

    if not enabled:
        global _LANGFUSE_AUTO_DISABLED
        _LANGFUSE_AUTO_DISABLED = True
        _remove_callback(litellm.success_callback, _LANGFUSE_CALLBACK)
        _remove_callback(litellm.failure_callback, _LANGFUSE_CALLBACK)
        return False

    _LANGFUSE_AUTO_DISABLED = False

    _set_env_if_provided("LANGFUSE_PUBLIC_KEY", public_key)
    _set_env_if_provided("LANGFUSE_SECRET_KEY", secret_key)
    _set_env_if_provided("LANGFUSE_HOST", host)
    _set_env_if_provided("LANGFUSE_RELEASE", release)
    _set_env_if_provided("LANGFUSE_DEBUG", debug)

    resolved_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    resolved_secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not resolved_public_key or not resolved_secret_key:
        if strict:
            raise ValueError(
                "Langfuse tracing requires LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY."
            )
        return False

    if importlib.util.find_spec("langfuse") is None:
        message = (
            "Langfuse tracing is configured, but the optional 'langfuse' package "
            "is not installed. Install it with `pip install datafast[langfuse]` "
            "or `pip install langfuse`."
        )
        if strict:
            raise RuntimeError(message)

        global _MISSING_LANGFUSE_WARNING_EMITTED
        if not _MISSING_LANGFUSE_WARNING_EMITTED:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            _MISSING_LANGFUSE_WARNING_EMITTED = True
        return False

    _append_unique(litellm.success_callback, _LANGFUSE_CALLBACK)
    _append_unique(litellm.failure_callback, _LANGFUSE_CALLBACK)
    return True


def maybe_configure_langfuse_tracing(*, load_env: bool = True) -> bool:
    """Best-effort Langfuse activation from the current environment."""
    if _LANGFUSE_AUTO_DISABLED:
        return False
    if is_langfuse_tracing_enabled():
        return True
    return configure_langfuse_tracing(load_env=load_env, strict=False)


def build_trace_metadata(
    *,
    model: Any,
    component: str,
    metadata: dict[str, Any] | None = None,
    trace_name: str | None = None,
    session_id: str | None = None,
    step_name: str | None = None,
    step_type: str | None = None,
    record_index: int | None = None,
    prompt_index: int | None = None,
    output_index: int | None = None,
    language_code: str | None = None,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Build consistent LiteLLM metadata for downstream observability."""
    base: dict[str, Any] = {
        "datafast_component": component,
        "datafast_version": _DATAFAST_VERSION,
        "datafast_provider": _get_model_attr(model, "provider_name"),
        "datafast_model_id": _get_model_attr(model, "model_id"),
    }

    if trace_name:
        base["trace_name"] = trace_name
    if session_id:
        base["session_id"] = session_id
    if step_name:
        base["datafast_step"] = step_name
    if step_type:
        base["datafast_step_type"] = step_type
    if record_index is not None:
        base["datafast_record_index"] = record_index
    if prompt_index is not None:
        base["datafast_prompt_index"] = prompt_index
    if output_index is not None:
        base["datafast_output_index"] = output_index
    if language_code:
        base["datafast_language"] = language_code
    if call_id:
        base["datafast_call_id"] = call_id

    if metadata:
        base.update(metadata)

    return base


def _append_unique(values: list[Any], value: Any) -> None:
    if value not in values:
        values.append(value)


def _remove_callback(values: list[Any], value: Any) -> None:
    while value in values:
        values.remove(value)


def _set_env_if_provided(name: str, value: str | bool | None) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        os.environ[name] = "true" if value else "false"
        return
    os.environ[name] = value


def _get_model_attr(model: Any, attr_name: str) -> str:
    value = getattr(model, attr_name, None)
    if value is None:
        return model.__class__.__name__.lower()
    return str(value)


__all__ = [
    "build_trace_metadata",
    "configure_langfuse_tracing",
    "is_langfuse_tracing_enabled",
    "load_env_once",
    "maybe_configure_langfuse_tracing",
]
