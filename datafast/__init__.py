"""Datafast - composable pipeline steps for synthetic data generation."""

import importlib.metadata

from datafast.core.checkpoint import CheckpointManager, PipelineChangedError
from datafast.core.config import RunConfig, LLMExecutionStrategy
from datafast.core.runner import Runner, run_pipeline
from datafast.core.step import Pipeline, Step
from datafast.core.types import Record
from datafast.llm.provider import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    MistralProvider,
    OpenRouterProvider,
    OllamaProvider,
    openai,
    anthropic,
    gemini,
    mistral,
    openrouter,
    ollama,
)
from datafast.logger_config import configure_logger
from datafast.sinks.sink import Sink, JSONLSink, CSVSink, ListSink, ParquetSink, HubSink
from datafast.sources.seed import Seed, SeedDimension
from datafast.sources.source import Source, HuggingFaceSource
from datafast.tracing import (
    configure_langfuse_tracing,
    is_langfuse_tracing_enabled,
)
from datafast.transforms.branch import Branch, JoinBranches
from datafast.transforms.data_ops import Map, FlatMap, Filter, Group, Pair, Concat, Join
from datafast.transforms.llm_eval import Classify, Score, Compare
from datafast.transforms.llm_extract import Extract
from datafast.transforms.llm_step import LLMStep
from datafast.transforms.llm_transform import Rewrite
from datafast.transforms.sample import Sample

try:
    __version__ = importlib.metadata.version("datafast")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


def get_version() -> str:
    """Return the installed version of the datafast package."""
    return __version__


__all__ = [
    "Record",
    "Step",
    "Pipeline",
    "RunConfig",
    "LLMExecutionStrategy",
    "Runner",
    "run_pipeline",
    "CheckpointManager",
    "PipelineChangedError",
    "Source",
    "HuggingFaceSource",
    "Seed",
    "SeedDimension",
    "Sample",
    "Map",
    "FlatMap",
    "Filter",
    "Group",
    "Pair",
    "Concat",
    "Join",
    "LLMStep",
    "Classify",
    "Score",
    "Compare",
    "Rewrite",
    "Extract",
    "Branch",
    "JoinBranches",
    "Sink",
    "JSONLSink",
    "CSVSink",
    "ListSink",
    "ParquetSink",
    "HubSink",
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "MistralProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "openai",
    "anthropic",
    "gemini",
    "mistral",
    "openrouter",
    "ollama",
    "configure_logger",
    "configure_langfuse_tracing",
    "get_version",
    "is_langfuse_tracing_enabled",
]
