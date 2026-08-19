# Specialized LLM Steps

`Classify`, `Score`, `Compare`, `Rewrite` and `Extract` are LLM steps for five common
jobs. Each one writes the prompt for you and parses the answer into named columns, so you
declare the job instead of the wording.

`LLMStep` is the general one: you give it the prompt and say how to parse the answer.
Reach for a specialized step when your job is one of these five, and for `LLMStep` when it
is not.

## At a glance

| Step | What it does | Reads | Writes |
|---|---|---|---|
| `Classify` | picks labels from a fixed set | `input_columns` | `output_column`, default `"label"` |
| `Score` | gives a number inside a range | `input_columns` | `output_column`, default `"score"` |
| `Compare` | judges two columns against each other | `column_a`, `column_b` | `output_column`, default `"comparison"` |
| `Rewrite` | rewrites text in a chosen way | `input_column` | `{input_column}_rewritten` |
| `Extract` | pulls structured values out of text | `input_column` | one dict column, or one column per value |

Each of them also adds a `_model` column naming the served model that answered.

## Shared parameters

These are keyword arguments on all five steps, apart from the two exceptions the table
notes.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `llm` | `ServedModel \| list[ServedModel] \| Sample` | `None` | the served model, or several; required on `Rewrite` |
| `prompt` | `str \| None` | `None` | replaces the built-in prompt; `Rewrite` has none |
| `system_prompt` | `str \| None` | `None` | a system message put before the user message |
| `forward_columns` | `list[str] \| None` | `None` | keep only these input columns in the output |
| `exclude_columns` | `list[str] \| None` | `None` | drop these input columns from the output |
| `on_parse_error` | `str` | `"skip"` | `"skip"` drops the record, `"raise"` stops the run |

Passing a list of served models, or a `Sample` of them, runs **every** record against
**every** served model. Ten records and three served models is thirty records out, and
thirty LLM calls.

The placeholders `prompt` accepts differ per step:

| Step | Placeholders in `prompt` |
|---|---|
| `Classify`, `Score` | `{content}` — the input columns, joined |
| `Extract` | `{content}`, `{fields}` |
| `Compare` | `{text_a}`, `{text_b}`, `{criteria}`, `{column_a}`, `{column_b}`, plus any column |
| `Rewrite` | takes no `prompt` at all; use `custom_instruction` |

## An LLM or a plain function

`Classify`, `Score`, `Compare` and `Extract` also run on a plain Python function. Pass
`fn` instead of `llm`; passing both, or neither, raises `ValueError` when you build the
step.

| Step | What `fn` receives | What `fn` must return |
|---|---|---|
| `Classify` | the record | a label, or a list of labels when `multi_label=True` |
| `Score` | the record | a number |
| `Compare` | the record | `{"winner": "a"}` (optionally `score_a`, `score_b`, `reasoning`), or just `"a"` |
| `Extract` | the record | a `dict` of values |

```python
from datafast import Classify

step = Classify(
    labels=["short", "long"],
    input_columns=["text"],
    output_column="length_class",
    fn=lambda r: "short" if len(r["text"]) < 100 else "long",
)
```

Use `fn` when the answer is a rule, not a judgement: a length cutoff, a regular
expression, a lookup table. It costs nothing, it is exact, and it is repeatable. Mixing
both in one pipeline is normal — cheap rules first, the LLM only on what is left.

`Rewrite` has no `fn`. It always calls an LLM, and `llm` is required.

### `Classify`

Assigns one label, or several, from a set you fix up front.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `labels` | `list[str]` | required | the allowed labels |
| `input_columns` | `list[str]` | required | columns to classify |
| `output_column` | `str` | `"label"` | where the label goes |
| `multi_label` | `bool` | `False` | allow several labels instead of exactly one |
| `include_explanation` | `bool` | `False` | also write `{output_column}_explanation` |
| `include_confidence` | `bool` | `False` | also write `{output_column}_confidence`, a float |
| `labels_description` | `dict[str, str] \| None` | `None` | a description per label, shown in the prompt |
| `fn` | `Callable \| None` | `None` | classify with a function instead of an LLM |

```python
from datafast import Classify, openai

step = Classify(
    labels=["politics", "sports", "tech"],
    input_columns=["article"],
    output_column="topics",
    multi_label=True,
    labels_description={"tech": "Software, hardware, the internet"},
    llm=openai("gpt-4o-mini"),
)
```

`labels_description` may cover some labels only; the rest appear with their name alone.

The two label modes handle a bad answer differently. With `multi_label=False` an
unknown label is **written to the record anyway**, after a warning. With
`multi_label=True` unknown labels are **dropped**, which can leave an empty list. Check
the column against `labels` before you trust it.

### `Score`

Gives a number inside a range, against criteria you describe.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `input_columns` | `list[str]` | required | columns to score |
| `output_column` | `str` | `"score"` | where the score goes |
| `score_range` | `tuple[float, float]` | `(1, 10)` | the lowest and highest allowed score |
| `criteria` | `str \| None` | `None` | what to judge, in words |
| `rubric` | `dict[int, str] \| None` | `None` | what given scores mean |
| `include_explanation` | `bool` | `False` | also write `{output_column}_explanation` |
| `fn` | `Callable \| None` | `None` | score with a function instead of an LLM |

```python
from datafast import Score, openai

step = Score(
    input_columns=["question", "answer"],
    output_column="quality",
    score_range=(1, 5),
    criteria="helpfulness and accuracy",
    rubric={1: "Wrong or unhelpful", 3: "Adequate", 5: "Excellent"},
    llm=openai("gpt-4o-mini"),
)
```

`criteria` is one sentence of guidance. `rubric` anchors particular numbers; you do not
have to describe every number, only the ones worth pinning. Keys are shown in ascending
order.

The score is always forced into `score_range`. A model answering `99` on a `(1, 5)`
scale is stored as `5`, not rejected — so an out-of-range answer looks like a top score.
The same clamping applies to what `fn` returns.

### `Compare`

Judges two columns of the same record against each other. The usual use is preference
data: two candidate answers, which is better.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `column_a` | `str` | required | the first column to compare |
| `column_b` | `str` | required | the second column to compare |
| `criteria` | `str` | required | what to compare them on |
| `output_column` | `str` | `"comparison"` | prefix for the result columns |
| `output_mode` | `str` | `"winner"` | how much detail to write, see below |
| `score_range` | `tuple[float, float]` | `(1, 10)` | range for the two scores |
| `fn` | `Callable \| None` | `None` | compare with a function instead of an LLM |

`output_mode` decides which columns you get. Any other value raises `ValueError`.

| `output_mode` | Columns written |
|---|---|
| `"winner"` | `{output_column}` — `"a"`, `"b"` or `"tie"` |
| `"scores"` | the above, plus `{output_column}_score_a` and `{output_column}_score_b` |
| `"detailed"` | the above, plus `{output_column}_reasoning` |

```python
from datafast import Compare, openai

step = Compare(
    column_a="response_chosen",
    column_b="response_rejected",
    criteria="helpfulness and accuracy",
    output_mode="detailed",
    llm=openai("gpt-4o-mini"),
)
```

`score_range` is only used by `"scores"` and `"detailed"`; under `"winner"` it does
nothing. As with `Classify`, a winner that is not `"a"`, `"b"` or `"tie"` is written to
the record anyway, after a warning.

### `Rewrite`

Rewrites text in one of eight ways. It writes plain text, not JSON, so the answer goes
into the column as it comes back.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `input_column` | `str` | required | column holding the text to rewrite |
| `llm` | `ServedModel \| list[ServedModel] \| Sample` | required | the served model, or several |
| `output_column` | `str \| None` | `None` | where the new text goes; `{input_column}_rewritten` when `None` |
| `mode` | `str` | `"paraphrase"` | the kind of rewrite, see below |
| `preserve` | `list[str] \| None` | `None` | aspects the rewrite must keep, e.g. `["technical terms"]` |
| `num_variations` | `int` | `1` | how many rewrites per record |
| `custom_instruction` | `str \| None` | `None` | the instruction for `mode="custom"` |
| `target_audience` | `str \| None` | `None` | who to write for, for `mode="audience"` |
| `target_length` | `str \| None` | `None` | how long, for `mode="length"` |

| `mode` | What it does | Companion argument |
|---|---|---|
| `"paraphrase"` | different words, same meaning | — |
| `"simplify"` | simpler language, shorter sentences | `target_audience`, optional |
| `"formalize"` | formal, professional tone | — |
| `"informalize"` | casual, conversational tone | — |
| `"elaborate"` | more detail, examples, explanation | — |
| `"custom"` | whatever you write | `custom_instruction`, **required** |
| `"audience"` | rewritten for a named audience | `target_audience`, **required** |
| `"length"` | longer or shorter, e.g. `"half"`, `"2x"` | `target_length`, **required** |

The three required companions are checked when you build the step: `mode="custom"`
without `custom_instruction` raises `ValueError` immediately, not mid-run. An unknown
`mode` raises too.

```python
from datafast import Rewrite, openai

step = Rewrite(
    input_column="text",
    llm=openai("gpt-4o-mini"),
    mode="custom",
    custom_instruction="Rewrite as a single question",
    preserve=["technical terms"],
    num_variations=3,
)
```

`num_variations=3` makes three LLM calls per record and yields **three records**, each
with a `_variation` column holding `0`, `1` or `2`. That column only appears when
`num_variations` is above `1`.

### `Extract`

Pulls named values out of free text.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `input_column` | `str` | required | column holding the text |
| `fields` | `dict[str, str] \| None` | `None` | your own values: name → description |
| `extractor` | `str \| None` | `None` | a preset instead of `fields`, see below |
| `fn` | `Callable \| None` | `None` | extract with a function instead of an LLM |
| `flatten` | `bool` | `False` | one column per value, instead of one dict column |
| `output_column` | `str` | `"extracted"` | the dict column, when `flatten=False` |

Give exactly one of `fields`, `extractor` or `fn`. Zero or two raises `ValueError`.

`fields` maps a name to a description. The name becomes the column, the description tells
the model what to put there. Use it when you know what you want.

| `extractor` | Values it produces |
|---|---|
| `"entities"` | `persons`, `organizations`, `locations`, `dates` |
| `"facts"` | `facts` |
| `"keywords"` | `keywords` |
| `"metadata"` | `title`, `author`, `date`, `source`, `language` |
| `"summary_fields"` | `title`, `summary`, `key_points` |
| `"topics"` | `topics` |

```python
from datafast import Extract, openai

step = Extract(
    input_column="product_description",
    fields={
        "product_name": "The name of the product",
        "price": "Price in dollars, number only",
    },
    flatten=True,
    llm=openai("gpt-4o-mini"),
)
```

`flatten` decides the shape of the output. With `flatten=False`, the whole result is one
dict in `output_column`. With `flatten=True`, each value becomes its own column and
`output_column` is ignored. Flatten when the next step reads the values as columns, or
when you write to CSV; keep the dict when you only want to carry the result along.

A value the model leaves out is written as an empty string `""`, even for values that are
normally lists. An `"entities"` result with no organizations gives `""`, not `[]`.

## Things worth knowing

- **`fn` mode ignores `forward_columns` and `exclude_columns`** in `Classify`, `Score` and
  `Compare`, and adds no `_model` column. It copies the whole record and adds its result.
  `Extract` is the exception: its `fn` mode honours both.
- **`fn` results are not checked against `labels`.** `Classify` in `fn` mode writes
  whatever the function returns. `Score` in `fn` mode is still clamped to `score_range`.
- **`on_parse_error="skip"` skips every error, not just parse errors.** A timeout, a
  refusal or a bad key drops that record with a warning, and you end a run with fewer
  records than you started. Use `"raise"` while you are developing.
- **A bad label or winner is still written.** Only `multi_label=True` filters. Validate
  the column before training on it.
- **`Score` clamps instead of failing**, so an out-of-range answer becomes the top or
  bottom of `score_range`.
- **Several served models multiply your records**, one output record per record per
  served model. So does `num_variations` on `Rewrite`.
- **`Extract` fills missing values with `""`**, including where you expected a list.

## Where to go next

- [LLM step](llm_step.md) — `LLMStep`, the general one, and parse modes.
- [Served Models](served_models.md) — building the `llm` argument.
- [Sources & Seed](sources_and_seed.md) — where the records come from.
- [Glossary](../glossary.md) — the precise meaning of record, column, step and served model.
- [API reference](../api.md) — generated signatures for everything on this page.
