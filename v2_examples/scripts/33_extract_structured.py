"""Extract step: pull structured information from unstructured text.

Demonstrates:
  Part A — Custom fields extraction (LLM) with flatten=True
  Part B — Predefined extractor ("entities") (LLM)
  Part C — Function-based extraction (regex, no LLM)
"""

import re

from datafast_v2 import Source, Extract, Sink, openrouter
import litellm
litellm.suppress_debug_info = True

# model = ollama("gemma3:4b")
model = openrouter("mistralai/ministral-14b-2512", temperature=0.3)

# --- Part A: Custom fields, flattened into separate columns ---
pipeline_custom = (
    Source.list([
        {"description": "iPhone 15 Pro - $999 - Features: A17 chip, titanium frame, 48MP camera"},
        {"description": "Samsung Galaxy S24 - $799 - Features: Snapdragon 8 Gen 3, AI tools, 200MP camera"},
    ])
    >> Extract(
        input_column="description",
        fields={
            "product_name": "The name of the product",
            "price": "Price in dollars (number only)",
            "features": "List of key features as a JSON array of strings",
        },
        flatten=True,
        llm=model,
    )
    >> Sink.jsonl("v2_examples/outputs/33_extract_custom.jsonl")
)
pipeline_custom.run()

# --- Part B: Predefined "entities" extractor ---
pipeline_entities = (
    Source.list([
        {"article": "Apple CEO Tim Cook announced new products at the Cupertino headquarters on September 9, 2024."},
        {"article": "The European Union and Google reached an agreement in Brussels regarding AI regulation."},
    ])
    >> Extract(
        input_column="article",
        extractor="entities",
        llm=model,
    )
    >> Sink.jsonl("v2_examples/outputs/33_extract_entities.jsonl")
)
pipeline_entities.run()

# --- Part C: Function-based extraction (regex, no LLM) ---
pipeline_fn = (
    Source.list([
        {"text": "Contact us at info@example.com or visit https://example.com for details."},
        {"text": "Email support@test.org or alice@corp.io for help."},
    ])
    >> Extract(
        input_column="text",
        fn=lambda r: {
            "emails": re.findall(r"[\w.]+@[\w.]+", r["text"]),
            "urls": re.findall(r"https?://\S+", r["text"]),
        },
        flatten=True,
    )
    >> Sink.list()
)
pipeline_fn.run()
