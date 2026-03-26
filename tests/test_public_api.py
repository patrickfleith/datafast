from datafast import (
    Branch,
    Classify,
    Compare,
    Concat,
    configure_langfuse_tracing,
    Extract,
    Filter,
    FlatMap,
    Group,
    is_langfuse_tracing_enabled,
    Join,
    JoinBranches,
    JSONLSink,
    LLMStep,
    ListSink,
    Map,
    Pair,
    Pipeline,
    Rewrite,
    RunConfig,
    Sample,
    Score,
    Seed,
    Sink,
    Source,
    get_version,
    ollama,
    openrouter,
)


def test_top_level_exports_support_basic_pipeline_composition():
    sink = ListSink()

    pipeline = (
        Source.list([{"value": 1}, {"value": 2}])
        >> Map(lambda r: {**r, "double": r["value"] * 2})
        >> sink
    )

    assert isinstance(pipeline, Pipeline)

    output = pipeline.run()

    assert output == [
        {"value": 1, "double": 2},
        {"value": 2, "double": 4},
    ]
    assert sink.records == output


def test_factory_exports_are_available(monkeypatch):
    import datafast.llms as llms_module

    monkeypatch.setattr(llms_module, "load_env_once", lambda: None)
    monkeypatch.setattr(
        llms_module,
        "maybe_configure_langfuse_tracing",
        lambda load_env=False: False,
    )
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    model = ollama(model_id="gemma3:4b")

    assert model.model_id == "gemma3:4b"
    assert openrouter is not None
    assert Source is not None
    assert Sink is not None
    assert Seed is not None
    assert Sample is not None
    assert Map is not None
    assert FlatMap is not None
    assert Filter is not None
    assert Group is not None
    assert Pair is not None
    assert Concat is not None
    assert Join is not None
    assert LLMStep is not None
    assert Classify is not None
    assert Score is not None
    assert Compare is not None
    assert Rewrite is not None
    assert Extract is not None
    assert Branch is not None
    assert JoinBranches is not None
    assert JSONLSink is not None
    assert RunConfig is not None
    assert configure_langfuse_tracing is not None
    assert isinstance(get_version(), str)
    assert is_langfuse_tracing_enabled() is False
