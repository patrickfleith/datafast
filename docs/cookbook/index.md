# Cookbook

Each cookbook connects a script that you can run to a page that explains it.

## Cookbooks

<div class="grid cards showcase" markdown>

-   **[Text classification](text_classification.md)**

    Trail reports in 2 languages. Each report agrees with 1 label.

-   **[Personas](persona_generation.md)**

    Personas from real articles, then a related persona for each.

-   **[Domain text corpus](space_text_generation.md)**

    Text about spacecraft engineering in 2 languages.

-   **[Preference pairs](preference_with_scoring.md)**

    Chosen and rejected answers with scores. Pairs that are too close are removed.

</div>

## How a cookbook works

The Python script is the reference. Each cookbook page tells you:

- where the script is
- which inputs the script uses
- which prompt files the script uses
- where the script writes its output files

## Example datasets

Datafast steps are modular, and you can put them together in many different ways. Thus
the cookbooks show only some of the datasets that you can make. The
[example scripts](examples.md) make other datasets, for example:

- [Q&A pairs](examples.md#full-pipelines): questions for each domain and difficulty
  level, each with its answer. Script `38`.
- [Quality-filtered text](examples.md#full-pipelines): a judge gives each fact a score
  from 1 to 10. Scores of 7 and higher stay. Script `39`.
- [Classification data with verified labels](examples.md#full-pipelines): examples with
  labels, then `Classify` examines each label. Script `40`.
- [Augmented text](examples.md#full-pipelines): sentences with labels, and 2 paraphrases
  of each sentence. Script `41`.
- [Structured extraction](examples.md#specialized-llm-steps): names, prices and features
  that the LLM finds in product text. Script `33`.
- [Rewrite styles](examples.md#branching): the same text in many styles, merged into 1
  record. Script `35`.
