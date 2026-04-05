from datafast import LLMStep, ListSink, Source


def test_runner_passes_llm_messages_by_keyword():
    class FakeModel:
        provider_name = "fake"
        model_id = "fake-model"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def generate(
            self,
            prompt=None,
            messages=None,
            metadata=None,
            response_format=None,
        ):
            self.calls.append({
                "prompt": prompt,
                "messages": messages,
                "metadata": metadata,
            })
            return "done"

    model = FakeModel()
    sink = ListSink()

    pipeline = (
        Source.list([{"topic": "robotics"}])
        >> LLMStep(
            prompt="Write one short line about {topic}.",
            input_columns=["topic"],
            output_column="result",
            model=model,
        ).as_step("generate_copy")
        >> sink
    )

    output = pipeline.run()

    assert output == [{"topic": "robotics", "result": "done", "_model": "fake-model"}]
    assert len(model.calls) == 1
    assert model.calls[0]["prompt"] is None
    assert model.calls[0]["messages"] == [
        {"role": "user", "content": "Write one short line about robotics."}
    ]
