# Preference data with scoring

Build a preference dataset — one question, a good answer, a weak answer — and keep only
the pairs where the difference is real.

Preference data is what you train a reward model or run DPO on. Every row is a question
with two answers, one marked chosen and one rejected. Collecting that from humans is slow
and expensive, and generating it is easy to do badly: ask a model twice and you get two
answers that are equally good, which teaches nothing.

This recipe generates both halves on purpose. One prompt asks for an expert answer, the
other asks for a vague one, and the same question feeds both. Then it scores each answer
and keeps only the pairs the scores agree are different. The result is a dataset where the
preference is measured rather than assumed.

It is also the recipe with the most architecture per line: it is the one that needs
`Branch`, because the two answers are two paths over the same record.

- **Script:** `examples/scripts/42_pipeline_preference_with_scoring.py`
- **Output:** `examples/outputs/42_preference_dataset.jsonl`
- **Checkpoints:** `examples/checkpoints/42_preference`
- **3 rows** by default, and **five LLM calls per row**

## The pipeline

Eight steps: generate, split, merge, score twice, filter, write.

```python
from datafast import Branch, Filter, JoinBranches, LLMStep, Score, Seed, Sink

pipeline = (
    Seed.product(Seed.values("topic", ["gravity", "photosynthesis", "machine learning"]))
    >> LLMStep(
        prompt="Generate a thoughtful educational question about {topic}.",
        input_columns=["topic"],
        output_column="question",
        model=model,
    ).as_step("generate_question")
    >> Branch(
        chosen=LLMStep(...),      # the expert answer
        rejected=LLMStep(...),    # the vague answer
    ).as_step("branch_responses")
    >> JoinBranches()
    >> Score(...).as_step("score_chosen")
    >> Score(...).as_step("score_rejected")
    >> Filter(fn=lambda r: r.get("score_chosen", 0) - r.get("score_rejected", 0) >= 2)
       .as_step("filter_margin")
    >> Sink.jsonl(OUTPUT_PATH)
)
```

Record counts along the way:

| Step | In | Out | Why |
|---|---|---|---|
| `SeedSource` | 0 | 3 | one record per topic |
| `generate_question` | 3 | 3 | one question each |
| `branch_responses` | 3 | 6 | every record goes down both paths |
| `JoinBranches` | 6 | 3 | the two paths merge back into one record |
| `score_chosen` | 3 | 3 | adds `score_chosen` |
| `score_rejected` | 3 | 3 | adds `score_rejected` |
| `filter_margin` | 3 | ≤ 3 | drops pairs that are too close |
| `JSONLSink` | 3 | 3 | writes the file, passes records through |

`Seed.values` builds the three starting records — there is only one dimension here, so
the seed is a list, not a grid — and `Sink.jsonl` writes the file at the end. The doubling
at the branch and the halving at the join are the whole point: a `Branch` turns one record
into one per path, and `JoinBranches` puts them back together.
[Branching](../reference/branching.md) is the reference for both — this page only covers
what this recipe does with them.

## One question, two answers

```python
Branch(
    chosen=LLMStep(
        prompt=(
            "You are an expert. Answer this question thoroughly and accurately.\n\n"
            "Question: {question}"
        ),
        input_columns=["question"],
        output_column="response",
        model=model,
        system_prompt="You are a world-class educator. Be precise, detailed, and helpful.",
    ),
    rejected=LLMStep(
        prompt="Answer briefly and vaguely: {question}",
        input_columns=["question"],
        output_column="response",
        model=model,
    ),
)
```

Both paths are ordinary `LLMStep`s, both read `question`, and both write `response`. Only
the instructions differ, and the difference is deliberately blunt: an expert answering
thoroughly against an answer that is brief and vague.

The chosen path also carries a `system_prompt` where the rejected one has none. That is a
second lever pulling the same way — a system prompt shapes every answer on that path,
where the user prompt shapes only this one.

The **path names are the design**: `chosen` and `rejected` become the column suffixes at
the join, so the naming the training format expects falls out of the pipeline rather than
being renamed at the end.

One model does all of it here, which keeps the comparison about the prompts. Giving each
path its own model turns the same pipeline into a model comparison — see
[making it yours](#making-it-yours) below, and read the warning about `_model` first.

### What the join gives you

`JoinBranches()` groups the two records back by `_branch_id` and merges them:

- `topic` and `question` existed **before** the branch, so they are copied once, unsuffixed.
- `response` was **added** by each path, so it becomes `response_chosen` and
  `response_rejected`.
- The `_branch_*` bookkeeping columns are dropped.

No `suffixes` argument is passed, so each path falls back to `_{path name}`. That is why
the columns come out named after the paths.

!!! warning "The `_model` column after a join is not what it looks like"

    `_model` exists **before** the branch — `generate_question` stamped it — and both
    paths write it again. `JoinBranches` treats it as a pre-branch column and copies it
    once, unsuffixed, from the first path that produced a record. There is no
    `_model_chosen` or `_model_rejected`.

    With one model everywhere, as the script has it, nothing is lost. Give the paths
    different models and the merged row names only the first path's, silently. The two
    `Score` steps then overwrite it again, so the `_model` that reaches the file is the
    scoring model's. If you need to know which model wrote which answer, copy `_model`
    into a path-specific column inside each path, with a `Map`, before the join.

## Scoring both sides

```python
Score(
    input_columns=["question", "response_chosen"],
    output_column="score_chosen",
    score_range=(1, 10),
    llm=model,
    criteria="helpfulness, accuracy, and completeness",
).as_step("score_chosen")
```

`Score` is an [LLM step](../reference/llm_specialized.md) that asks for one number.
`llm=` takes the served model, `output_column` names the column the number lands in, and
`score_range` is the scale. It builds its own prompt from what you pass — you never write
it:

```text
Score the following content on a scale of 1 to 10.

Criteria: helpfulness, accuracy, and completeness

Content:
question: Why does gravity bend light?
response_chosen: An answer.

Respond with valid JSON: {"score": <number between 1 and 10>}
Return only the JSON object, no additional text.
```

Two things about that prompt are worth noticing. Several `input_columns` are rendered as
`name: value` lines, so the model sees which text is the question and which is the answer.
And each `Score` step is given **one** answer, not both: the two are scored independently,
against the criteria, rather than compared against each other. If you want a head-to-head
judgement instead, `Compare` is the step for it.

Scoring twice means two more calls per row, which is why one row costs five: one question,
two answers, two scores.

### What a score actually is

The number that lands in the column is not always the number the model said. Measured
against `score_range=(1, 10)`:

| The model replies | The column gets |
|---|---|
| `{"score": 7}` | `7.0` |
| `{"score": 99}` | `10` — clamped to the top of the range |
| `{"score": "high"}` | `1` — the bottom of the range |
| `{"explanation": "..."}` with no score | `1` — the bottom of the range |
| anything that is not JSON | nothing; the record is dropped |

Only the last row is an error. The other three are silent, and two of them turn a model
that ignored the instruction into a confident score — a perfect 10 or a bottom 1. On a
preference dataset that is worse than a dropped row, because the margin filter downstream
then keeps or discards the pair for a reason that has nothing to do with the answers.

Watch for it the cheap way: a histogram of `score_chosen` and `score_rejected` with spikes
exactly at the ends of the range is the symptom.

Note also that an in-range score arrives as a **float** and a clamped one as the range's
own value, so the column can hold `7.0` and `10` in the same file.

## Keeping only clear preferences

```python
Filter(
    fn=lambda r: r.get("score_chosen", 0) - r.get("score_rejected", 0) >= 2
).as_step("filter_margin")
```

This is the step that makes the dataset worth training on. A pair where both answers
scored 6 and 5 is not a preference — it is noise with a label. The margin of 2 on a 1–10
scale is the threshold, and it is inclusive:

| `score_chosen` | `score_rejected` | Margin | Kept |
|---|---|---|---|
| 9 | 4 | 5 | yes |
| 7 | 5 | 2 | yes |
| 6 | 5 | 1 | no |
| missing | 3 | −3 | no |

`fn` gets the whole record and returns a bool, so the comparison is plain Python — no
operator syntax to learn. `.get(column, 0)` rather than `record[column]` is what makes the
last row of that table a dropped record instead of a `KeyError`: a `Score` that failed to
parse drops the record before this step, but a `Score` you later remove would not.

Raise the margin for a smaller, cleaner dataset; lower it for more rows and more noise.
Whatever you choose, count what survives — a filter that drops everything is a silent
empty file.

## A row

```json
{
  "_model": "mistralai/ministral-14b-2512",
  "topic": "gravity",
  "question": "Why does gravity bend light?",
  "response_chosen": "Gravity curves spacetime itself, and light follows...",
  "response_rejected": "Because gravity pulls on things, including light.",
  "score_chosen": 9.0,
  "score_rejected": 4.0
}
```

That is the standard preference shape: a prompt column and two response columns, plus the
scores that justified the pair. Rename `question` to `prompt` and the two response columns
to `chosen` and `rejected` with a `Map` if your trainer expects those names.

## Running it

```bash
python examples/scripts/42_pipeline_preference_with_scoring.py
```

You need `OPENROUTER_API_KEY` in a `.env` file at the repository root. The script builds
its model with `openrouter(...)`; the commented-out `ollama("gemma3:4b")` line above it
runs the whole recipe locally, for free, if you have Ollama.

```python
records = pipeline.run(
    checkpoint_dir="examples/checkpoints/42_preference",
    batch_size=2,
)
```

The checkpoint directory holds one file per step, and the branch gets one per path:

```text
manifest.json
step_000_SeedSource.jsonl
step_001_generate_question.jsonl
step_002_branch_responses.chosen.jsonl
step_002_branch_responses.rejected.jsonl
step_002_branch_responses.jsonl
step_003_JoinBranches.jsonl
step_004_score_chosen.jsonl
step_005_score_rejected.jsonl
step_006_filter_margin.jsonl
step_007_JSONLSink.jsonl
```

!!! note "Checkpoints are written, not read"

    `resume` defaults to `False`, and this script does not set it. The files above are
    written on every run and never used, so running the script twice pays for all fifteen
    calls twice. Add `resume=True` to the `run()` call to reuse them.

Only four of the eight steps are named. The rest show up as their class names —
`SeedSource`, `JoinBranches`, `JSONLSink` — which is fine until you want to
`resume_from` one of them. Name the steps you might restart from.

`batch_size=2` sets how many LLM calls go out per batch. Note what it costs when one
fails: a failing call takes its whole batch group with it, so a larger batch loses more.
See [Error handling & troubleshooting](../guides/troubleshooting.md).

## Making it yours

| To change | Edit | Effect |
|---|---|---|
| the subject matter | the `topic` values | linear on the row count |
| how bad the rejected answer is | the `rejected` path's prompt | changes the margin, not the count |
| what "good" means | `criteria` on both `Score` steps | keep the two identical, or the scores are not comparable |
| how strict the scoring is | `rubric=` on the `Score` steps | anchors the scale to descriptions instead of a bare 1–10 |
| why a score was given | `include_explanation=True` | adds a `{output_column}_explanation` column |
| how different the pair must be | the margin in `filter_margin` | higher keeps fewer, cleaner pairs |

Three extensions worth the effort:

- **A model per path.** Put a strong model on `chosen` and a small one on `rejected` and
  the dataset compares models instead of prompts. Copy `_model` inside each path first,
  as the warning above explains.
- **A third path.** `Branch` takes any number of paths, so a `middle` path gives you
  ranked triples rather than pairs. The join suffixes it the same way.
- **Deduplicate the questions.** Three topics produce three questions; a hundred topics
  will produce near-duplicates. `Group` on a normalized `question` finds them before you
  pay for four calls each.

## Where to go next

- [Branching](../reference/branching.md) — `Branch`, `JoinBranches`, suffixes and the join rules.
- [Specialized LLM steps](../reference/llm_specialized.md) — `Score`, `Compare`, `Classify`, `Rewrite`, `Extract`.
- [Data ops](../reference/data_ops.md) — `Filter`, `Map`, `Group` and the rest.
- [Pipelines & execution](../guides/pipelines_and_execution.md) — checkpoints, resume and run controls.
- [Text classification](text_classification.md) — a simpler recipe, seed-driven end to end.
