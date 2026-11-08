# Cookbook

Cookbooks connect a runnable script to a documentation walkthrough.

## Example datasets

Eight kinds of dataset you can build with datafast today. Each square links to its
cookbook walkthrough, or to the example script that builds it.

<div class="grid cards showcase" markdown>

-   **[Q&A pairs](examples.md#full-pipelines)**

    Questions by domain and difficulty, each with its answer.

    Script `38`

-   **[Text classification](text_classification.md)**

    Multilingual trail reports, each written to fit one label.

    Cookbook

-   **[Personas](persona_generation.md)**

    Personas inferred from real articles, then expanded.

    Cookbook

-   **[Domain text corpus](space_text_generation.md)**

    Spacecraft engineering text in several languages.

    Cookbook

-   **[Preference pairs](preference_with_scoring.md)**

    Chosen and rejected answers, scored, close pairs dropped.

    Cookbook

-   **[Quality-filtered text](examples.md#full-pipelines)**

    Facts scored 1 to 10 by a judge, 7 and up kept.

    Script `39`

-   **[Augmented text](examples.md#full-pipelines)**

    Labelled sentences plus two paraphrases of each.

    Script `41`

-   **[Structured extraction](examples.md#specialized-llm-steps)**

    Names, prices and features pulled out of product text.

    Script `33`

</div>

## How a cookbook works

The Python script is the source of truth. Each cookbook page explains:

- where the executable example lives
- what inputs it uses
- which prompt assets it depends on
- where it writes its output artifacts

## Available Cookbooks

- [Examples](examples.md): all 45 runnable scripts in `examples/scripts/`, grouped by what they show.
- [Text Classification](text_classification.md): generate a multilingual trail-conditions classification dataset from explicit seed dimensions.
- [Persona Generation](persona_generation.md): infer personas from real articles and expand them through relationships using randomized prompt variants.
- [Space Engineering Text Generation](space_text_generation.md): generate a raw multilingual technical text corpus from seed dimensions.
- [Preference Data with Scoring](preference_with_scoring.md): generate chosen/rejected answer pairs with `Branch`, score both sides, and keep only clear preferences.
