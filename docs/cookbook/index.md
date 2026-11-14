# Cookbook

Each cookbook is an end-to-end example of how datafast works for a specific dataset example.

## Cookbooks

<div class="grid cards showcase" markdown>

-   **[Text classification](text_classification.md)**

    Generate synthetic trail reports from hikers in multiple languages with classification labels.
    
    Script `45`

-   **[Personas](persona_generation.md)**

    Generate synthetic personas from real article seeds.

    Script `43`

-   **[Domain text corpus](space_text_generation.md)**

    Generate text content about spacecraft engineering in multiple languages.

    Script `44`

-   **[Preference pairs](preference_with_scoring.md)**

    Make chosen and rejected answer pairs with scores for RLHF.

    Script `42`

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

## Built with datafast badge

If you publish a dataset that you made with datafast, you can add this badge to its
dataset card:

[![Built with Datafast](https://raw.githubusercontent.com/patrickfleith/datafast/main/assets/datafast-badge-web.png)](https://github.com/patrickfleith/datafast)

```markdown
[![Built with Datafast](https://raw.githubusercontent.com/patrickfleith/datafast/main/assets/datafast-badge-web.png)](https://github.com/patrickfleith/datafast)
```
