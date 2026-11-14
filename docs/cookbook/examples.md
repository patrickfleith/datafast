# Examples

The `examples/scripts/` directory has 45 scripts that you can run. The numbers show the
order to read them in. The first scripts do not use an LLM. Each script in the middle
shows 1 feature. The last scripts make full datasets.

Run each script from the root of the repository. Each script writes to
`examples/outputs/`:

```bash
python examples/scripts/01_seed_product_basic.py
```

**Scripts 01 to 14 do not use an API key.** They make records, change them and write
files. They are the cheapest way to learn how a pipeline works.

Scripts 15 to 45 call a model through [OpenRouter](../reference/providers/openrouter.md).
Put `OPENROUTER_API_KEY` in a `.env` file at the root of the repository. Refer to
[Installation](../installation.md).

Most LLM scripts have an `ollama(...)` line in a comment, above the model line. To use a
local model at no cost, remove the comment marker from the `ollama(...)` line. Then put a
comment marker on the other model line.

## Seeds and sources: no API key

These scripts show how to start a pipeline. A seed makes records from dimensions. A
source loads records from a different location.

| Script | What it shows |
|---|---|
| `01_seed_product_basic.py` | `Seed.values` and `Seed.product`: 3 topics × 2 difficulty levels × 2 languages = 12 records |
| `02_seed_zip.py` | `Seed.zip` makes pairs of dimension values in sequence, not all combinations |
| `03_seed_expand_and_range.py` | `Seed.expand` for parent and child pairs, and `Seed.range` for numbers, in combination |
| `04_seed_with_sampling.py` | `Sample` decreases 192 combinations to 20, with the uniform and stratified strategies |
| `05_source_list_and_file.py` | `Source.list` and `Source.jsonl` in, JSONL and Parquet out |

## Data ops: no API key

These scripts show the transforms that change records between the source and the sink.
Each script shows 1 part of [Data ops](../reference/data_ops.md).

| Script | What it shows |
|---|---|
| `06_map_transform.py` | `Map`: 1 record in, 1 record out |
| `07_flatmap_explode.py` | `FlatMap`: 1 record in, many records out, from a list column |
| `08_filter_declarative.py` | `Filter` with `where` operators: `$gte`, `$in`, `$contains`, `$len_gt`, `$or` and `$and` |
| `09_filter_fn_and_drop.py` | `Filter(fn=...)` for conditions that the operators cannot show, and `keep=False` to do the opposite |
| `10_sample_strategies.py` | the 9 `Sample` strategies, for comparison |
| `11_group_aggregate.py` | `Group` with `collect`, aggregate functions and `min_per_group` |
| `12_pair_records.py` | `Pair` with the random, sliding and sequential strategies |
| `13_concat_sources.py` | `Concat` puts the records of many pipelines into 1 sequence |
| `14_join_datasets.py` | `Join` on a shared key: inner, left, right and outer |

## LLM steps

Each script shows 1 `LLMStep` feature. The scripts show most of
[LLM step](../reference/llm_step.md).

| Script | What it shows |
|---|---|
| `15_llm_text_mode.py` | the default: the full answer becomes 1 column |
| `16_llm_json_mode.py` | `parse_mode="json"` divides 1 answer into many columns |
| `17_llm_xml_mode.py` | `parse_mode="xml"` does the same with tags |
| `18_llm_multi_prompt.py` | a list of prompts: each record goes through each prompt and gets `_prompt_index` |
| `19_llm_multi_model.py` | a list of models: each model generates each record, and the record gets `_model` |
| `20_llm_languages.py` | `language={"code": "Name"}` multiplies the records by the number of languages and adds `_language` |
| `21_llm_num_outputs.py` | `num_outputs=3` makes 3 different generations for each record |
| `22_llm_system_prompt.py` | `system_prompt` sets the same voice for all calls in the step |
| `23_llm_forward_exclude.py` | `forward_columns` and `exclude_columns` select the output columns |
| `24_llm_prompt_from_file.py` | a prompt template in a `.txt` file, not in the pipeline |
| `25_llm_skip_if.py` | `skip_if` prevents the call for records that do not have a use for it |
| `26_llm_sample_in_config.py` | `Sample` in a step: select a prompt for each record, or a model 1 time |

## Specialized LLM steps

These steps ask for 1 specific result and parse it for you. Refer to
[Specialized LLM steps](../reference/llm_specialized.md). Each step has an `fn` mode that
does not use a model. Some scripts show the 2 modes.

| Script | What it shows |
|---|---|
| `27_classify_llm.py` | `Classify` with labels, an explanation and a confidence |
| `28_classify_multilabel_and_fn.py` | classification with many labels, then the same task with a usual function |
| `29_score_llm_and_fn.py` | `Score` with criteria, a rubric and an explanation, then a score from a function |
| `30_compare_responses.py` | `Compare` judges 2 columns against each other |
| `31_rewrite_paraphrase.py` | `Rewrite` in paraphrase mode, with many variants for each record |
| `32_rewrite_modes.py` | the other `Rewrite` modes: simplify, formalize, audience and custom |
| `33_extract_structured.py` | `Extract` with custom fields, a predefined extractor and a regex function |

## Branching

These scripts send each record through 2 branch paths, then merge the results. Refer to
[Branching](../reference/branching.md).

| Script | What it shows |
|---|---|
| `34_branch_preference_data.py` | `Branch` and `JoinBranches` for `response_chosen` and `response_rejected` |
| `35_branch_rewrite_styles.py` | many `Rewrite` styles at the same time, merged with custom suffixes |

## Running a pipeline

These scripts show the functions of the runner. Refer to
[Pipelines & execution](../guides/pipelines_and_execution.md).

| Script | What it shows |
|---|---|
| `36_checkpointing_and_resume.py` | `checkpoint_dir`, `resume=True` and why steps must have names |
| `37_limit_stop_and_strategies.py` | `limit`, `stop_after` and the 3 `llm_strategy` orders |

## Full pipelines

These scripts put many steps together to make a dataset that you can keep.

| Script | What it shows |
|---|---|
| `38_pipeline_qa_generation.py` | seeds, then a question, then an answer, in JSON mode, with checkpoints |
| `39_pipeline_generate_score_filter.py` | generate, score, then keep only the records with good scores |
| `40_pipeline_classification_dataset.py` | generate examples with labels, then examine the labels with `Classify` |
| `41_pipeline_text_augmentation.py` | `Rewrite` to make more data, and `Concat` to keep the original and the new text |
| `42_pipeline_preference_with_scoring.py` | branch, score the 2 answers, then filter on the margin |

## Cookbooks

These 4 scripts have a page on this site. Read the page together with the script.

| Script | What it makes | Page |
|---|---|---|
| `42_pipeline_preference_with_scoring.py` | pairs of chosen and rejected answers, with scores and a filter | [Preference data with scoring](preference_with_scoring.md) |
| `43_cookbook_persona_generation.py` | personas from real articles, then a related persona for each | [Persona generation](persona_generation.md) |
| `44_cookbook_space_text_generation.py` | a technical text corpus in 2 languages, from a seed | [Space engineering text generation](space_text_generation.md) |
| `45_cookbook_text_classification.py` | a classification dataset with labels, in 2 languages | [Text classification](text_classification.md) |

Scripts 43, 44 and 45 can push to the Hugging Face Hub. For a push, do
`pip install "datafast[hub]"` and get a token. Script 43 also reads a Hugging Face
dataset as its source. Thus you must install the same extra, also when you do not publish.

**Before you run these scripts, set `HF_REPO_ID` to a repository that you own.**

## Related pages

- [Quickstart](../quickstart.md): the shortest full pipeline.
- [Key Concepts](../concepts.md): records, steps and the runner.
- [Guides](../guides/index.md): the same subjects as the scripts, with more explanation.
