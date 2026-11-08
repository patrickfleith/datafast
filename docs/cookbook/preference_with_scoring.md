# Preference data with scoring

Make a preference dataset: 1 question, 1 good answer and 1 bad answer. Keep only the
pairs with a clear difference.

You use preference data to train a reward model or to do DPO. Each record has a question
and 2 answers. One answer is "chosen" and the other answer is "rejected". To get this
data from people is slow and expensive. To generate it correctly is difficult. If you ask
a model 2 times, you usually get 2 answers of the same quality. A model cannot learn from
these pairs.

This procedure generates the 2 answers with different instructions. One prompt asks for
the answer of an expert. The other prompt asks for an unclear answer. The 2 prompts get
the same question. Then the pipeline gives a score to each answer. It keeps only the
pairs where the scores show a difference. Thus the dataset measures each preference.

This procedure uses `Branch`, because the 2 answers are 2 branch paths on the same record.

- **Script:** `examples/scripts/42_pipeline_preference_with_scoring.py`
- **Output:** `examples/outputs/42_preference_dataset.jsonl`
- **Checkpoints:** `examples/checkpoints/42_preference`
- **3 records** by default, with **5 LLM calls for each record**

## The pipeline

The pipeline has 8 steps: generate, divide, merge, score 2 times, filter, write.

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

The record count after each step:

| Step | In | Out | Why |
|---|---|---|---|
| `SeedSource` | 0 | 3 | 1 record for each topic |
| `generate_question` | 3 | 3 | 1 question for each record |
| `branch_responses` | 3 | 6 | each record goes through the 2 branch paths |
| `JoinBranches` | 6 | 3 | the 2 branch paths merge into 1 record |
| `score_chosen` | 3 | 3 | adds `score_chosen` |
| `score_rejected` | 3 | 3 | adds `score_rejected` |
| `filter_margin` | 3 | ≤ 3 | removes the pairs that are too close |
| `JSONLSink` | 3 | 3 | writes the file and sends the records through |

`Seed.values` makes the 3 initial records. The seed has only 1 dimension, so it is a
list. `Sink.jsonl` writes the file at the end.

The record count becomes 2 times larger at the branch and 2 times smaller at the join.
`Branch` makes 1 record for each branch path from each record. `JoinBranches` merges the
records again. [Branching](../reference/branching.md) is the reference for the 2 steps.
This page tells you only how this procedure uses them.

## One question and 2 answers

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

The 2 branch paths are usual `LLMStep`s. The 2 steps read `question` and write
`response`. Only the instructions are different, and the difference is large: a full
answer from an expert, and a short, unclear answer.

The `chosen` branch path also has a `system_prompt`. The `rejected` branch path has no
system prompt. The system prompt has an effect on each answer in its branch path. The user
prompt has an effect only on 1 answer. Thus the system prompt makes the difference
between the 2 answers larger.

**The branch path names are important.** At the join, `chosen` and `rejected` become the
suffixes of the columns. Thus the pipeline gives the names that the training format
expects. You do not change the names at the end.

One model does all the work here. Thus the comparison is only between the prompts. If you
give a different model to each branch path, the pipeline compares models. Refer to
[Change the dataset](#change-the-dataset) and read the warning about `_model` first.

### The result of the join

`JoinBranches()` makes groups of the 2 records with `_branch_id` and merges them:

- `topic` and `question` were in the record **before** the branch. The join copies them
  1 time, without a suffix.
- Each branch path **adds** `response`. Thus the join makes `response_chosen` and
  `response_rejected`.
- The join removes the `_branch_*` columns.

The step has no `suffixes` argument, so each branch path uses `_{branch path name}`. Thus
the columns get the names of the branch paths.

!!! warning "After a join, the `_model` column can be incorrect"

    `_model` is in the record **before** the branch, because `generate_question` adds it.
    The 2 branch paths then write it again. `JoinBranches` thinks that it is a column from
    before the branch. Thus it copies the column 1 time, without a suffix, from the first
    branch path that gives a record. There is no `_model_chosen` and no
    `_model_rejected`.

    If all steps use 1 model, as in the script, you lose no data. If the branch paths use
    different models, the merged record shows only the model of the first branch path,
    and you get no warning. The 2 `Score` steps then write `_model` again. Thus the
    `_model` in the file is the scoring model. If you must know which model wrote each
    answer, add a `Map` in each branch path. The `Map` copies `_model` to a column for
    that branch path, before the join.

## Score the 2 answers

```python
Score(
    input_columns=["question", "response_chosen"],
    output_column="score_chosen",
    score_range=(1, 10),
    llm=model,
    criteria="helpfulness, accuracy, and completeness",
).as_step("score_chosen")
```

`Score` is an [LLM step](../reference/llm_specialized.md) that asks for 1 number:

- `llm=` takes the served model.
- `output_column` gives the column for the number.
- `score_range` gives the scale.

`Score` makes its prompt from these arguments. You do not write it:

```text
Score the following content on a scale of 1 to 10.

Criteria: helpfulness, accuracy, and completeness

Content:
question: Why does gravity bend light?
response_chosen: An answer.

Respond with valid JSON: {"score": <number between 1 and 10>}
Return only the JSON object, no additional text.
```

Know 2 facts about this prompt:

- The prompt shows each column in `input_columns` as a `name: value` line. Thus the model
  can see which text is the question and which text is the answer.
- Each `Score` step gets **1** answer. It does not get the 2 answers. Thus each answer
  gets a score against the criteria, and not against the other answer. To compare the 2
  answers directly, use `Compare`.

The 2 scores are 2 more calls for each record. Thus each record uses 5 calls: 1
question, 2 answers and 2 scores.

### How `Score` sets the value

The value in the column is not always the number from the model. With
`score_range=(1, 10)`:

| The model answers | The column gets |
|---|---|
| `{"score": 7}` | `7.0` |
| `{"score": 99}` | `10`, the maximum of the range |
| `{"score": "high"}` | `1`, the minimum of the range |
| `{"explanation": "..."}` without a score | `1`, the minimum of the range |
| an answer that is not JSON | no value. The step removes the record |

Only the last line is an error. The other 3 lines give no warning. In 2 of them, a model
that did not obey the instruction gets a score of 10 or 1. In a preference dataset, this
is worse than a removed record. The margin filter then keeps or removes the pair for a
reason that is not related to the answers.

To find this problem, make a histogram of `score_chosen` and `score_rejected`. Many
values at the 2 ends of the range show the problem.

Also, a score in the range is a **float**, and a score at a limit is the value of that
limit. Thus the same file can contain `7.0` and `10` in the column.

## Keep only clear preferences

```python
Filter(
    fn=lambda r: r.get("score_chosen", 0) - r.get("score_rejected", 0) >= 2
).as_step("filter_margin")
```

This step makes the dataset useful for training. If the 2 answers get 6 and 5, the pair
is not a preference. It is noise with a label. The margin is 2 on a scale of 1 to 10.
The filter keeps a pair with a margin of exactly 2:

| `score_chosen` | `score_rejected` | Margin | Kept |
|---|---|---|---|
| 9 | 4 | 5 | yes |
| 7 | 5 | 2 | yes |
| 6 | 5 | 1 | no |
| missing | 3 | −3 | no |

`fn` gets the full record and returns a bool. Thus the comparison is usual Python, with
no operator syntax.

The step uses `.get(column, 0)` and not `record[column]`. Thus a missing score removes
the record, as in the last line of the table, and does not cause a `KeyError`. Now, a
`Score` that cannot parse an answer removes the record before this step. But if you
remove a `Score` step later, this protection becomes necessary.

For a smaller dataset with less noise, increase the margin. For more records and more
noise, decrease it. After the run, count the records that stay. If the filter removes all
records, the file is empty, and you get no warning.

## A record

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

This is the usual preference shape: 1 prompt column and 2 answer columns. The scores
show why the pipeline kept the pair. If your trainer expects different names, use a `Map`.
Change `question` to `prompt`, and change the 2 answer columns to `chosen` and
`rejected`.

## Run the script

```bash
python examples/scripts/42_pipeline_preference_with_scoring.py
```

Put `OPENROUTER_API_KEY` in a `.env` file at the root of the repository. The script makes
its served model with `openrouter(...)`. Above that line, there is an `ollama("gemma3:4b")`
line in a comment. If you have Ollama, use that line to run all the procedure on your
computer, at no cost.

```python
records = pipeline.run(
    checkpoint_dir="examples/checkpoints/42_preference",
    batch_size=2,
)
```

The checkpoint directory has 1 file for each step. The branch has 1 file for each branch
path:

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

!!! note "The runner writes the checkpoints but does not read them"

    The default value of `resume` is `False`, and this script does not set it. The runner
    writes the files above at each run, but it does not use them. Thus, if you run the
    script 2 times, it makes the 15 calls 2 times. To use the checkpoints, add
    `resume=True` to the `run()` call.

Only 4 of the 8 steps have a name. The other steps use their class names: `SeedSource`,
`JoinBranches` and `JSONLSink`. This is not a problem until you must use `resume_from`
on one of them. Give a name to each step that you can start again from.

`batch_size=2` sets the number of LLM calls in each batch. If 1 call fails, all the
calls in its batch group fail. Thus a larger batch loses more calls. Refer to
[Error handling & troubleshooting](../guides/troubleshooting.md).

## Change the dataset

| To change | Edit | Effect |
|---|---|---|
| the subjects | the `topic` values | linear on the record count |
| the quality of the rejected answer | the prompt of the `rejected` branch path | changes the margin, not the count |
| the meaning of "good" | `criteria` on the 2 `Score` steps | use the same criteria on the 2 steps. If not, you cannot compare the scores |
| how strict the scores are | `rubric=` on the `Score` steps | gives a description for each value of the scale, not only 1 to 10 |
| the reason for a score | `include_explanation=True` | adds a `{output_column}_explanation` column |
| the necessary difference in a pair | the margin in `filter_margin` | a higher margin keeps fewer pairs, with less noise |

You can add these changes:

- **A model for each branch path.** Put a strong model on `chosen` and a small model on
  `rejected`. The dataset then compares models, not prompts. First copy `_model` in each
  branch path, as the warning above tells you.
- **A third branch path.** `Branch` accepts 1 or more branch paths. A `middle` branch
  path gives you sets of 3 ranked answers, not pairs. The join adds the suffix in the
  same way.
- **Remove duplicate questions.** Three topics give 3 questions. One hundred topics give
  questions that are almost the same. Use `Group` on a normalized `question` to find
  them before you make 4 more calls for each.

## Related pages

- [Branching](../reference/branching.md): `Branch`, `JoinBranches`, suffixes and the join rules.
- [Specialized LLM steps](../reference/llm_specialized.md): `Score`, `Compare`, `Classify`, `Rewrite` and `Extract`.
- [Data ops](../reference/data_ops.md): `Filter`, `Map`, `Group` and the other transforms.
- [Pipelines & execution](../guides/pipelines_and_execution.md): checkpoints, resume and run controls.
- [Text classification](text_classification.md): a simpler procedure that uses a seed from start to end.
