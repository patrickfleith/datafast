"""Sink steps for datafast."""

from datafast.sinks.sink import (
    Sink,
    JSONLSink,
    CSVSink,
    ListSink,
    ParquetSink,
    HubSink,
)

__all__ = ["Sink", "JSONLSink", "CSVSink", "ListSink", "ParquetSink", "HubSink"]
