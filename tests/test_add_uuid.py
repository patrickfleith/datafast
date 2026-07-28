import uuid

from datafast import AddUUID, LLMStep, Sink, Source


def assert_valid_uuid(value: str) -> None:
    parsed = uuid.UUID(value)
    assert str(parsed) == value


def test_add_uuid_adds_id_when_missing():
    records = list(AddUUID().process([{"text": "hello"}]))

    assert records[0]["text"] == "hello"
    assert_valid_uuid(records[0]["id"])


def test_add_uuid_preserves_existing_id_by_default():
    records = list(AddUUID().process([{"id": "source-1", "text": "hello"}]))

    assert records == [{"id": "source-1", "text": "hello"}]


def test_add_uuid_overwrites_existing_id_when_requested():
    records = list(
        AddUUID(overwrite=True).process([{"id": "source-1", "text": "hello"}])
    )

    assert records[0]["text"] == "hello"
    assert records[0]["id"] != "source-1"
    assert_valid_uuid(records[0]["id"])


def test_add_uuid_generates_distinct_ids_for_multiple_records():
    records = list(AddUUID().process([{"text": "a"}, {"text": "b"}]))
    ids = [record["id"] for record in records]

    assert len(set(ids)) == 2
    for value in ids:
        assert_valid_uuid(value)


def test_add_uuid_supports_custom_column_name():
    records = list(AddUUID(column="example_id").process([{"text": "hello"}]))

    assert "id" not in records[0]
    assert_valid_uuid(records[0]["example_id"])


def test_add_uuid_assigns_unique_ids_to_llm_num_outputs_pipeline():
    class FakeModel:
        model_id = "fake-model"
        provider_name = "fake"

        def generate(self, messages, metadata=None):
            return '{"title": "Generated", "text": "Body"}'

    pipeline = (
        Source.list([{"topic": "vacuum"}])
        >> LLMStep(
            prompt="Write about {topic}.",
            input_columns=["topic"],
            output_columns=["title", "text"],
            parse_mode="json",
            model=FakeModel(),
            num_outputs=2,
        )
        >> AddUUID()
        >> Sink.list()
    )

    records = pipeline.run()
    ids = [record["id"] for record in records]

    assert len(records) == 2
    assert len(set(ids)) == 2
    for value in ids:
        assert_valid_uuid(value)
