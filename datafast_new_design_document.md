# Datafast New Version — Architecture Design Document

> **Version:** 1.0  
> **Status:** Design Specification  
> **Author:** Patrick Fleith / Claude  
> **Date:** February 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [Core Concepts](#3-core-concepts)
4. [The Sample Abstraction](#4-the-sample-abstraction)
5. [Seed Builders](#5-seed-builders)
6. [Record and Data Flow](#6-record-and-data-flow)
7. [Step Taxonomy](#7-step-taxonomy)
8. [Source Steps](#8-source-steps)
9. [Sink Steps](#9-sink-steps)
10. [Transform Steps — Data Operations](#10-transform-steps--data-operations)
11. [Transform Steps — LLM Generation](#11-transform-steps--llm-generation)
12. [Transform Steps — LLM Evaluation](#12-transform-steps--llm-evaluation)
13. [Transform Steps — LLM Transformation](#13-transform-steps--llm-transformation)
14. [Combination Steps](#14-combination-steps)
15. [Pipeline Definition](#15-pipeline-definition)
16. [Execution Model](#16-execution-model)
17. [Checkpointing and Resumption](#17-checkpointing-and-resumption)
18. [Model Abstraction](#18-model-abstraction)
19. [Error Handling](#19-error-handling)
20. [Module Structure](#20-module-structure)
21. [Implementation Priorities](#21-implementation-priorities)
22. [API Quick Reference](#22-api-quick-reference)
23. [Future Extensions](#23-future-extensions)

---

## 1. Executive Summary

### 1.1 Purpose

Datafast 2.0 is a complete redesign of the synthetic data generation library. The goal is to move from **monolithic dataset classes** to **composable pipeline steps** that can be orchestrated to create any type of synthetic dataset.

### 1.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pipelines over Dataset Classes** | More flexible, composable, reusable steps |
| **Step-by-step execution** | Complete each step before the next; enables checkpointing, inspection, and optimization |
| **Unified Sample abstraction** | Same sampling concept for data records and LLM config (prompts, models) |
| **Single value vs list convention** | Intuitive: `model=gpt4` uses one model; `model=[gpt4, claude]` uses all |
| **Declarative pipeline, imperative execution** | Pipeline describes *what*; execution config describes *how* |
| **Checkpoint after every step** | Robustness; resume on failure; inspect intermediate results |

### 1.3 What Changes from Datafast 1.0

| Datafast 1.0 | Datafast 2.0 |
|--------------|--------------|
| `ClassificationDataset`, `MCQDataset`, etc. | Pipelines composed of generic steps |
| Hardcoded generation loops | Configurable execution with batching, parallelism |
| Single LLM per dataset | Multiple LLMs per step, different LLMs per step |
| Limited checkpointing | Checkpoint after every step |
| Fixed dataset schemas | Dynamic records, flexible schemas |

---

## 2. Design Philosophy

### 2.1 Principle: Composition over Configuration

Instead of:
```python
# Old: Monolithic class with many config options
MCQDataset(
    hf_dataset_name="...",
    num_samples_per_prompt=3,
    distractor_prompt="...",
    ...
)
```

We have:
```python
# New: Pipeline of composable steps
(
    Source.huggingface("...")
    >> LLMStep(prompt="Generate questions...", num_outputs=3, ...)
    >> LLMStep(prompt="Generate distractors...", ...)
    >> Sink.jsonl("mcq.jsonl")
)
```

### 2.2 Principle: Explicit over Implicit

Every operation is a visible step in the pipeline. No hidden magic inside dataset classes.

### 2.3 Principle: Pythonic Conventions

- **Single value** = use that value
- **List of values** = use all values (cartesian product)
- **Sample(...)** = select from values using a strategy

No need to learn new abstractions like "Sampler protocols." Just pass values or lists.

### 2.4 Principle: Fail Early, Resume Gracefully

- Validate pipeline structure before execution
- Checkpoint after every step
- Clear error messages with context
- Resume from any checkpoint

### 2.5 Principle: Inspect Anywhere

- View intermediate results after any step
- Test with a single record before scaling
---

## 3. Core Concepts

### 3.1 Glossary

| Term | Definition |
|------|------------|
| **Record** | A single data item (Python dict with string keys and any values) |
| **Step** | A transformation that takes records and produces records |
| **Pipeline** | A directed acyclic graph (DAG) of steps |
| **Source** | A step that produces records from external data (no input) |
| **Sink** | A step that consumes records and writes to external storage (no output) |
| **Sample** | An abstraction for selecting from a collection |
| **Seed** | Configuration-driven generation of initial records |

### 3.2 Record Structure

Records are Python dictionaries or possibly even better: Pydantic objects

```python
record = {
    "id": "doc_001",
    "text": "The quick brown fox...",
    "category": "nature",
    "score": 0.85,
    "tags": ["animals", "classics"],
    "metadata": {"source": "wikipedia", "year": 2024},
}
```

Records can have any fields. Steps add, remove, or modify fields.

### 3.3 Pipeline as DAG

A pipeline is a directed acyclic graph where:
- Nodes are steps
- Edges represent data flow
- Most pipelines are linear chains
- Branch/JoinBranches creates parallel paths

```
Linear:     Source >> Step1 >> Step2 >> Step3 >> Sink

Branching:  Source >> Step1 >> Branch(a=Step2a, b=Step2b) >> JoinBranches >> Step3 >> Sink

            Which is equivalent to:
            
                              ┌─► Step2a ─┐
            Source >> Step1 ──┤           ├──► Step3 >> Sink
                              └─► Step2b ─┘
```

---

## 4. The Sample Abstraction

### 4.1 Purpose

`Sample` is a unified way to select items from a collection. It works for:
- **Data records** — select which records to process
- **LLM config** — select which prompts, models, or parameters to use

### 4.2 Core Principle

| What you write | What it means |
|----------------|---------------|
| `value` | Use this single value |
| `[a, b, c]` | Use all values (exhaustive) |
| `Sample([a, b, c], n=2)` | Select 2 values using a strategy |
| `Sample(...).pick()` | Select once upfront, then use selected values exhaustively |

### 4.3 Sample Class Definition

```python
class Sample:
    """Select items from a collection using various strategies."""
    
    def __init__(
        self,
        items: list | Iterable,
        
        # How many to select
        n: int | None = None,              # absolute count
        frac: float | None = None,         # fraction (0.0 to 1.0)
        
        # Selection strategy
        strategy: str = "uniform",
        
        # Strategy-specific parameters
        by: str | Callable | list[float] | None = None,
        ascending: bool = False,
        center: float | None = None,
        std: float | None = None,
        bins: int | None = None,
        
        # Reproducibility
        seed: int | None = None,
        
        # Behavior
        replace: bool = False,             # sample with replacement
    ):
        ...
    
    def pick(self, n: int | None = None) -> list:
        """
        Materialize the sample immediately and return a plain list.
        Useful when you want to sample once upfront, then use the 
        selected items exhaustively.
        """
        ...
    
    def __iter__(self):
        """Allow iteration over all items (for exhaustive use)."""
        return iter(self.items)
    
    def __len__(self):
        return len(self.items)
```

### 4.4 Sampling Strategies

#### For Data Records

| Strategy | Description | Required `by` | Example |
|----------|-------------|---------------|---------|
| `"uniform"` | Equal probability random selection | No | `Sample(n=100)` |
| `"first"` | First N records | No | `Sample(n=100, strategy="first")` |
| `"last"` | Last N records | No | `Sample(n=100, strategy="last")` |
| `"systematic"` | Every Nth record | No, uses `step` param | `Sample(strategy="systematic", step=10)` |
| `"top"` | Highest values | Column name or function | `Sample(n=100, strategy="top", by="score")` |
| `"bottom"` | Lowest values | Column name or function | `Sample(n=100, strategy="top", by="score", ascending=True)` |
| `"weighted"` | Probability proportional to value | Column name or function | `Sample(n=100, strategy="weighted", by="score")` |
| `"stratified"` | Maintain category distribution | Categorical column | `Sample(n=100, strategy="stratified", by="category")` |
| `"gaussian"` | Weight by normal distribution | Numeric column + `center`, `std` | `Sample(n=100, strategy="gaussian", by="length", center=500, std=100)` |
| `"diverse"` | Maximize variety (embedding-based) | Text column | `Sample(n=100, strategy="diverse", by="text")` |

#### For Config Items (prompts, models)

| Strategy | Description | Required `by` | Example |
|----------|-------------|---------------|---------|
| `"uniform"` | Equal probability | No | `Sample(prompts, n=2)` |
| `"weighted"` | Weighted by explicit weights | List of weights | `Sample(models, n=1, strategy="weighted", by=[0.5, 0.3, 0.2])` |
| `"first"` | First N items | No | `Sample(prompts, n=3, strategy="first")` |

### 4.5 Sample Usage Examples

#### Sampling Data Records (as a pipeline step)

```python
# 100 random records
Source.huggingface("dataset") >> Sample(n=100)

# 10% random sample
Source.huggingface("dataset") >> Sample(frac=0.1)

# First 50 records (for testing)
Source.huggingface("dataset") >> Sample(n=50, strategy="first")

# Top 100 by score
Source.huggingface("dataset") >> Sample(n=100, strategy="top", by="score")

# Stratified by category
Source.huggingface("dataset") >> Sample(n=500, strategy="stratified", by="category")

# Favor documents around 1000 words
Source.huggingface("dataset") >> Sample(
    n=200, 
    strategy="gaussian", 
    by=lambda r: len(r["text"].split()),
    center=1000,
    std=200,
)

# Maximize diversity
Source.huggingface("dataset") >> Sample(n=100, strategy="diverse", by="text")
```

#### Sampling Config (within LLMStep)

```python
# All prompts (exhaustive)
LLMStep(prompt=["P1", "P2", "P3"], ...)  # 3 prompts per record

# 2 random prompts per record
LLMStep(prompt=Sample(["P1", "P2", "P3"], n=2), ...)

# 1 model, weighted selection
LLMStep(
    model=Sample([gpt4, claude, gemini], n=1, strategy="weighted", by=[0.5, 0.3, 0.2]),
    ...
)

# Sample once upfront, use those for all records
selected_prompts = Sample(all_prompts, n=5, seed=42).pick()
LLMStep(prompt=selected_prompts, ...)  # same 5 prompts for all records
```

### 4.6 Implementation Notes

- `Sample` used as a pipeline step operates on the data flow
- `Sample` used in LLMStep parameters operates on config per-record
- When `Sample` is in a parameter, sampling happens **per input record** unless `.pick()` was called
- The `diverse` strategy requires computing embeddings; this should be done lazily and cached

---

## 5. Seed Builders

### 5.1 Purpose

Seeds create initial records from configuration, not from existing data. Use when you don't have a dataset to start with, but rather structured config like:

- Topics and subtopics
- Personas
- Languages
- Parameter combinations

### 5.2 Seed Class Definition

```python
class Seed:
    """Build initial records from configuration."""
    
    @staticmethod
    def values(column: str, values: list) -> SeedDimension:
        """
        Create a dimension with explicit values.
        
        Example:
            Seed.values("language", ["en", "fr", "de"])
            # Creates 3 potential records with {"language": "en"}, etc.
        """
        ...
    
    @staticmethod
    def expand(parent: str, child: str, mapping: dict[str, list]) -> SeedDimension:
        """
        Create a dimension from nested structure.
        
        Example:
            Seed.expand("topic", "subtopic", {
                "Physics": ["Quantum", "Relativity"],
                "Biology": ["Genetics", "Evolution"],
            })
            # Creates 4 potential records:
            # {"topic": "Physics", "subtopic": "Quantum"}
            # {"topic": "Physics", "subtopic": "Relativity"}
            # {"topic": "Biology", "subtopic": "Genetics"}
            # {"topic": "Biology", "subtopic": "Evolution"}
        """
        ...
    
    @staticmethod
    def range(column: str, start: int, end: int, step: int = 1) -> SeedDimension:
        """
        Create a dimension from numeric range.
        
        Example:
            Seed.range("grade_level", 1, 12)
            # Creates 12 potential records with grade_level 1 through 12
        """
        ...
    
    @staticmethod
    def product(*dimensions: SeedDimension) -> Source:
        """
        Create cartesian product of all dimensions.
        Returns a Source that can be used in a pipeline.
        
        Example:
            Seed.product(
                Seed.values("persona", ["student", "teacher"]),
                Seed.values("language", ["en", "fr"]),
            )
            # Creates 4 records (2 × 2)
        """
        ...
    
    @staticmethod
    def zip(*dimensions: SeedDimension) -> Source:
        """
        Zip dimensions together (must be same length).
        
        Example:
            Seed.zip(
                Seed.values("question", ["Q1", "Q2", "Q3"]),
                Seed.values("answer", ["A1", "A2", "A3"]),
            )
            # Creates 3 records: (Q1,A1), (Q2,A2), (Q3,A3)
        """
        ...
```

### 5.3 Seed Usage Examples

```python
# Simple: 3 topics × 2 languages = 6 seed records
seeds = Seed.product(
    Seed.values("topic", ["AI", "Climate", "Health"]),
    Seed.values("language", ["en", "fr"]),
)

# Nested: topics with subtopics
seeds = Seed.product(
    Seed.expand("domain", "topic", {
        "Science": ["Physics", "Chemistry", "Biology"],
        "Humanities": ["History", "Philosophy"],
    }),
    Seed.values("persona", ["student", "expert"]),
)
# Creates: 5 topics × 2 personas = 10 seed records

# With sampling
seeds = (
    Seed.product(
        Seed.expand("domain", "topic", large_topic_hierarchy),  # 100+ topics
        Seed.values("persona", personas),  # 10 personas
        Seed.values("language", ["en", "fr", "de", "es"]),  # 4 languages
    )
    >> Sample(n=200)  # don't exhaust all 4000+, just sample 200
)

# Use seeds in pipeline
pipeline = (
    seeds
    >> LLMStep(
        prompt="As a {persona}, ask a question about {topic} in {language}.",
        input_columns=["persona", "topic", "language"],
        output_columns=["question"],
        model=gpt4,
    )
    >> Sink.jsonl("questions.jsonl")
)
```

### 5.4 Implementation Notes

- `SeedDimension` is an internal type representing a single axis of variation
- `Seed.product()` returns a `Source` (or Source-compatible object) that can start a pipeline
- `Seed.zip()` requires all dimensions to have the same length; raises error otherwise
- Seeds are generated lazily when the pipeline executes

---

## 6. Record and Data Flow

### 6.1 Record Type

Records are Python dictionaries with string keys:

```python
from typing import Any

Record = dict[str, Any]
```

We explicitly avoid Pydantic models for records to keep them flexible. Steps can add any fields.

### 6.2 Data Flow Between Steps

Each step receives an iterable of records and yields an iterable of records:

```python
class Step:
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        ...
```

In step-by-step execution mode (our default), the entire iterable is consumed and materialized between steps. This enables:
- Checkpointing after each step
- Progress tracking
- Memory-efficient batch processing within each step

### 6.3 Record Identity

Records don't have a built-in identity field. If you need to track records through a pipeline (e.g., for joins), add an ID field explicitly:

```python
Source.list(documents)
>> Map(lambda r: {**r, "_id": uuid4().hex})
>> ...
```

Or use `forward_columns` in LLMStep to preserve identifying fields.

---

## 7. Step Taxonomy

### 7.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               STEP TAXONOMY                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SOURCES (0 → N records)                                                         │
│  ───────────────────────                                                         │
│  Source.huggingface    Load from HuggingFace Hub                                 │
│  Source.file           Load from local file (JSONL, CSV, Parquet, TXT)           │
│  Source.list           Load from Python list                                     │
│  Seed.product          Generate from configuration                               │
│                                                                                  │
│  SINKS (N → 0 records)                                                           │
│  ─────────────────────                                                           │
│  Sink.jsonl            Write to JSONL file                                       │
│  Sink.csv              Write to CSV file                                         │
│  Sink.parquet          Write to Parquet file                                     │
│  Sink.hub              Push to HuggingFace Hub                                   │
│  Sink.list             Collect to Python list                                    │
│                                                                                  │
│  DATA OPERATIONS (no LLM)                                                        │
│  ────────────────────────                                                        │
│  Map                   Transform each record (1:1)                               │
│  FlatMap               Transform each record to multiple (1:N)                   │
│  Filter                Keep/drop records by condition                            │
│  Sample                Select records using strategy                             │
│  Pair                  Create pairs/tuples within data                           │
│  Group                 Aggregate records by key                                  │
│                                                                                  │
│  LLM GENERATION (create new content)                                             │
│  ───────────────────────────────────                                             │
│  LLMStep               Free-form generation with custom prompt                   │
│                                                                                  │
│  LLM TRANSFORMATION (modify existing content)                                    │
│  ────────────────────────────────────────────                                    │
│  Rewrite               Paraphrase, simplify, formalize, etc.                     │
│  Extract               Pull structured fields from text                          │
│                                                                                  │
│  LLM EVALUATION (judge content)                                                  │
│  ──────────────────────────────                                                  │
│  Classify              Assign label(s) from fixed set                            │
│  Score                 Assign numeric score in range                             │
│  Compare               Pairwise comparison of two fields                         │
│                                                                                  │
│  MULTI-DATA OPERATIONS                                                           │
│  ─────────────────────                                                           │
│  Concat                Stack multiple data sources vertically                    │
│  Join                  Merge two data sources by key                             │
│                                                                                  │
│  BRANCHING                                                                       │
│  ─────────                                                                       │
│  Branch                Split into parallel paths                                 │
│  JoinBranches          Merge parallel paths back                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Step Base Class

```python
from abc import ABC, abstractmethod
from typing import Iterable

class Step(ABC):
    """Base class for all pipeline steps."""
    
    def __init__(self):
        self._name: str | None = None
    
    @abstractmethod
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """Process input records and yield output records."""
        ...
    
    def as_step(self, name: str) -> "Step":
        """Assign a name to this step for checkpointing and debugging."""
        self._name = name
        return self
    
    def __rshift__(self, other: "Step") -> "Pipeline":
        """Enable >> syntax for chaining steps."""
        return Pipeline([self, other])
    
    @property
    def name(self) -> str:
        """Return step name (auto-generated if not set)."""
        return self._name or self.__class__.__name__
```

---

## 8. Source Steps

### 8.1 Source.huggingface

Load data from HuggingFace Hub.

```python
class HuggingFaceSource(Step):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        columns: list[str] | None = None,  # select specific columns
        trust_remote_code: bool = False,
    ):
        ...

# Usage
Source.huggingface("wikipedia", split="train")
Source.huggingface("squad", split="validation", columns=["question", "context", "answers"])
```

### 8.2 Source.file

Load data from local files.

```python
class FileSource(Step):
    def __init__(
        self,
        path: str,
        format: str | None = None,  # auto-detect from extension if None
        # Format-specific options
        text_column: str = "text",  # for TXT files: column name for each line
        **kwargs,  # passed to underlying reader (pandas, etc.)
    ):
        ...

# Usage
Source.file("data.jsonl")
Source.file("data.csv")
Source.file("data.parquet")
Source.file("documents.txt", text_column="content")  # each line becomes a record
```

**Supported formats:**
- `.jsonl` / `.json` — JSON Lines or JSON array
- `.csv` — Comma-separated values
- `.tsv` — Tab-separated values
- `.parquet` — Apache Parquet
- `.txt` — Plain text (one record per line)

### 8.3 Source.list

Load data from a Python list.

```python
class ListSource(Step):
    def __init__(self, records: list[Record]):
        ...

# Usage
Source.list([
    {"text": "Document 1...", "category": "science"},
    {"text": "Document 2...", "category": "history"},
])
```

---

## 9. Sink Steps

### 9.1 Sink.jsonl

Write records to a JSONL file.

```python
class JSONLSink(Step):
    def __init__(
        self,
        path: str,
        append: bool = False,
        columns: list[str] | None = None,  # select specific columns to write
    ):
        ...

# Usage
Sink.jsonl("output.jsonl")
Sink.jsonl("output.jsonl", columns=["question", "answer", "score"])  # only these fields
```

### 9.2 Sink.csv

Write records to a CSV file.

```python
class CSVSink(Step):
    def __init__(
        self,
        path: str,
        columns: list[str] | None = None,
        **kwargs,  # passed to csv.DictWriter
    ):
        ...

# Usage
Sink.csv("output.csv")
```

### 9.3 Sink.parquet

Write records to a Parquet file.

```python
class ParquetSink(Step):
    def __init__(
        self,
        path: str,
        columns: list[str] | None = None,
    ):
        ...

# Usage
Sink.parquet("output.parquet")
```

### 9.4 Sink.hub

Push records to HuggingFace Hub as a dataset.

```python
class HubSink(Step):
    def __init__(
        self,
        repo_id: str,
        token: str | None = None,  # uses HF_TOKEN env var if None
        private: bool = True,
        train_size: float | None = None,  # if set, creates train/test split
        seed: int = 42,
        shuffle: bool = True,
        commit_message: str | None = None,
    ):
        ...

# Usage
Sink.hub("username/my-dataset")
Sink.hub("username/my-dataset", train_size=0.9, private=False)
```

### 9.5 Sink.list

Collect records into a Python list (for testing/inspection).

```python
class ListSink(Step):
    def __init__(self):
        self.records: list[Record] = []
    
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        self.records = list(records)
        return iter([])  # sink produces no output

# Usage
sink = Sink.list()
pipeline = Source.list(data) >> SomeStep() >> sink
pipeline.run()
print(sink.records)
```

---

## 10. Transform Steps — Data Operations

### 10.1 Map

Transform each record one-to-one.

```python
class Map(Step):
    def __init__(self, fn: Callable[[Record], Record]):
        self.fn = fn
    
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        for record in records:
            yield self.fn(record)

# Usage
>> Map(lambda r: {**r, "text_length": len(r["text"])})
>> Map(lambda r: {"id": r["id"], "content": r["text"].upper()})
```

### 10.2 FlatMap

Transform each record into zero or more records.

```python
class FlatMap(Step):
    def __init__(self, fn: Callable[[Record], list[Record]]):
        self.fn = fn
    
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        for record in records:
            yield from self.fn(record)

# Usage: Explode a list field
>> FlatMap(lambda r: [
    {"id": r["id"], "question": q} 
    for q in r["questions"]
])

# Usage: Duplicate with variations
>> FlatMap(lambda r: [
    {**r, "style": "formal"},
    {**r, "style": "casual"},
])
```

### 10.3 Filter

Keep or drop records based on conditions.

```python
class Filter(Step):
    def __init__(
        self,
        fn: Callable[[Record], bool] | None = None,
        where: dict | None = None,
        expr: str | None = None,
        keep: bool = True,  # if False, drop matches instead of keeping
    ):
        ...
    
    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        for record in records:
            match = self._evaluate(record)
            if match == self.keep:
                yield record

# Usage: Function-based
>> Filter(fn=lambda r: len(r["text"]) > 100)

# Usage: Declarative
>> Filter(where={"score": {"$gte": 7}, "category": "science"})
>> Filter(where={"text": {"$len_gt": 100, "$len_lt": 5000}})
>> Filter(where={"category": {"$in": ["science", "tech"]}})

# Usage: Expression
>> Filter(expr="score >= 7 AND category = 'science'")

# Usage: Drop instead of keep
>> Filter(where={"quality": {"$lt": 3}}, keep=False)  # remove low quality
```

**Declarative Operators:**

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals (default) | `{"status": "active"}` |
| `$ne` | Not equals | `{"status": {"$ne": "deleted"}}` |
| `$gt`, `$gte` | Greater than (or equal) | `{"score": {"$gt": 7}}` |
| `$lt`, `$lte` | Less than (or equal) | `{"score": {"$lt": 3}}` |
| `$in` | In list | `{"category": {"$in": ["a", "b"]}}` |
| `$nin` | Not in list | `{"category": {"$nin": ["spam"]}}` |
| `$contains` | String contains | `{"text": {"$contains": "AI"}}` |
| `$startswith` | String starts with | `{"text": {"$startswith": "The"}}` |
| `$endswith` | String ends with | `{"text": {"$endswith": "."}}` |
| `$regex` | Regex match | `{"email": {"$regex": "@.*\\.edu$"}}` |
| `$len_gt`, `$len_lt`, `$len_eq` | Length comparison | `{"text": {"$len_gt": 100}}` |
| `$exists` | Field is not None | `{"summary": {"$exists": True}}` |
| `$type` | Type check | `{"tags": {"$type": "list"}}` |
| `$all` | All values present (for lists) | `{"tags": {"$all": ["ml", "nlp"]}}` |
| `$any` | Any value present (for lists) | `{"tags": {"$any": ["ml", "nlp"]}}` |

**Logical Operators:**

```python
# AND (implicit when multiple conditions)
>> Filter(where={"score": {"$gte": 7}, "category": "science"})

# OR
>> Filter(where={"$or": [
    {"category": "science"},
    {"score": {"$gte": 9}},
]})

# AND with nested OR
>> Filter(where={"$and": [
    {"category": {"$in": ["science", "tech"]}},
    {"$or": [
        {"score": {"$gt": 7}},
        {"featured": True},
    ]},
]})
```

### 10.4 Sample (as Step)

Select records using a sampling strategy. (See Section 4 for full details.)

```python
# Usage as pipeline step
>> Sample(n=100)  # 100 uniform random
>> Sample(n=100, strategy="top", by="score")  # top 100 by score
>> Sample(n=100, strategy="stratified", by="category")  # maintain distribution
>> Sample(frac=0.1, strategy="diverse", by="text")  # 10% diverse sample
```

---

## 11. Transform Steps — LLM Generation

### 11.1 LLMStep

The core step for free-form LLM generation. This is the most flexible and commonly used step.

```python
class LLMStep(Step):
    def __init__(
        self,
        # === Prompt Configuration ===
        prompt: str | list[str] | Sample,
        input_columns: list[str],              # columns to inject into prompt
        output_columns: list[str],             # expected output fields
        
        # === Model Configuration ===
        model: Model | list[Model] | Sample,
        temperature: float | list[float] = 0.7,
        max_tokens: int = 1024,
        top_p: float | None = None,
        
        # === Output Multiplication ===
        num_outputs: int = 1,                  # outputs per input×prompt×model combo
        
        # === Column Handling ===
        forward_columns: list[str] | None = None,   # columns to keep from input
        exclude_columns: list[str] | None = None,   # columns to drop from input
        
        # === Language ===
        language: str | list[str] | dict[str, str] | Sample | None = None,
        # If dict, keys are codes, values are names: {"en": "English", "fr": "French"}
        # Adds {language} and {language_name} to prompt context
        
        # === Control ===
        skip_if: Callable[[Record], bool] | None = None,
        
        # === System Prompt ===
        system_prompt: str | None = None,
    ):
        ...
```

**How expansion works:**

For each input record, LLMStep generates outputs for:
```
all prompts × all models × all languages × num_outputs
```

If any parameter is a single value, it contributes ×1.
If any parameter is a list, it contributes ×len(list).
If any parameter is a Sample, it samples per record.

**Prompt Templating:**

Prompts use `{field_name}` syntax. Available variables:
- All `input_columns` from the record
- `{language}` — language code (if language is set)
- `{language_name}` — language name (if language is set)

**Output Structure:**

For each output, LLMStep creates a new record with:
- All `forward_columns` from input (or all input columns minus `exclude_columns`)
- All `output_columns` parsed from LLM response
- `_model` — which model generated this output
- `_prompt_index` — which prompt was used (if multiple prompts)
- `_language` — which language (if multiple languages)

**Usage Examples:**

```python
# Simple: one prompt, one model
>> LLMStep(
    prompt="Summarize this text: {text}",
    input_columns=["text"],
    output_columns=["summary"],
    model=gpt4,
)

# Multiple outputs per input
>> LLMStep(
    prompt="Generate a question about: {text}",
    input_columns=["text"],
    output_columns=["question", "answer"],
    model=gpt4,
    num_outputs=3,  # 3 Q&A pairs per input
)

# Multiple prompts (exhaustive)
>> LLMStep(
    prompt=[
        "Generate a factual question: {text}",
        "Generate an inferential question: {text}",
    ],
    input_columns=["text"],
    output_columns=["question", "answer"],
    model=gpt4,
)
# → 2 outputs per input (one per prompt)

# Multiple models (exhaustive)
>> LLMStep(
    prompt="Generate a question: {text}",
    input_columns=["text"],
    output_columns=["question"],
    model=[gpt4, claude],
)
# → 2 outputs per input (one per model)

# Multiple languages
>> LLMStep(
    prompt="Translate to {language_name}: {text}",
    input_columns=["text"],
    output_columns=["translation"],
    model=gpt4,
    language={"en": "English", "fr": "French", "de": "German"},
)
# → 3 outputs per input (one per language)

# Sampled prompts (per record)
>> LLMStep(
    prompt=Sample([prompt1, prompt2, prompt3, prompt4, prompt5], n=2),
    input_columns=["text"],
    output_columns=["question"],
    model=gpt4,
)
# → 2 outputs per input (2 randomly selected prompts)

# Full combination
>> LLMStep(
    prompt=["Prompt A: {text}", "Prompt B: {text}"],  # 2 prompts
    input_columns=["text"],
    output_columns=["question"],
    model=[gpt4, claude],                              # 2 models
    language={"en": "English", "fr": "French"},        # 2 languages
    num_outputs=2,                                     # 2 per combo
)
# → 2 × 2 × 2 × 2 = 16 outputs per input
```

---

## 12. Transform Steps — LLM Evaluation

### 12.1 Classify

Assign one or more labels from a fixed set.

```python
class Classify(Step):
    def __init__(
        self,
        labels: list[str],
        input_columns: list[str],
        
        output_column: str = "label",
        multi_label: bool = False,
        include_explanation: bool = False,  # adds {output_column}_explanation
        include_confidence: bool = False,   # adds {output_column}_confidence
        
        # Classification method (one of these)
        llm: Model | list[Model] | Sample | None = None,
        prompt: str | None = None,           # custom prompt (has defaults)
        fn: Callable[[Record], str | list[str]] | None = None,
        
        # For LLM-based
        labels_description: dict[str, str] | None = None,
    ):
        ...

# Usage: LLM-based sentiment
>> Classify(
    labels=["positive", "negative", "neutral"],
    input_columns=["review_text"],
    output_column="sentiment",
    llm=gpt4,
    include_explanation=True,
)

# Usage: Multi-label topics
>> Classify(
    labels=["politics", "sports", "tech", "entertainment"],
    input_columns=["article"],
    output_column="topics",
    multi_label=True,
    llm=gpt4,
    labels_description={
        "politics": "Government, elections, policy",
        "sports": "Athletic events, teams, players",
        "tech": "Technology, software, startups",
        "entertainment": "Movies, music, celebrities",
    },
)

# Usage: Function-based
>> Classify(
    labels=["short", "medium", "long"],
    input_columns=["text"],
    output_column="length_class",
    fn=lambda r: "short" if len(r["text"]) < 100 else "medium" if len(r["text"]) < 500 else "long",
)
```

### 12.2 Score

Assign a numeric score within a range.

```python
class Score(Step):
    def __init__(
        self,
        input_columns: list[str],
        
        output_column: str = "score",
        range: tuple[float, float] = (1, 10),
        include_explanation: bool = False,
        
        # Scoring method (one of these)
        llm: Model | list[Model] | Sample | None = None,
        prompt: str | None = None,
        fn: Callable[[Record], float] | None = None,
        
        # For LLM-based
        criteria: str | None = None,
        rubric: dict[int, str] | None = None,
    ):
        ...

# Usage: LLM-based quality scoring
>> Score(
    input_columns=["question", "answer"],
    output_column="quality",
    range=(1, 10),
    llm=gpt4,
    criteria="helpfulness, accuracy, and completeness",
    rubric={
        1: "Completely wrong or unhelpful",
        3: "Partially addresses the question with major issues",
        5: "Adequate response with some gaps",
        7: "Good response, minor improvements possible",
        10: "Excellent, comprehensive, accurate",
    },
    include_explanation=True,
)

# Usage: Function-based (e.g., readability)
>> Score(
    input_columns=["text"],
    output_column="readability",
    range=(0, 100),
    fn=lambda r: textstat.flesch_reading_ease(r["text"]),
)
```

### 12.3 Compare

Pairwise comparison of two fields.

```python
class Compare(Step):
    def __init__(
        self,
        column_a: str,
        column_b: str,
        criteria: str,
        
        output_column: str = "comparison",
        output_mode: str = "winner",  # "winner", "scores", "detailed"
        
        llm: Model | list[Model] | Sample | None = None,
    ):
        ...

# Usage: Compare two responses
>> Compare(
    column_a="response_chosen",
    column_b="response_rejected",
    criteria="helpfulness and accuracy",
    output_mode="detailed",
    llm=gpt4,
)
# Output: comparison = {
#   "winner": "a",  # or "b" or "tie"
#   "score_a": 8,
#   "score_b": 5,
#   "reasoning": "Response A provides more detail..."
# }
```

---

## 13. Transform Steps — LLM Transformation

### 13.1 Rewrite

Generate variations of text while preserving meaning.

```python
class Rewrite(Step):
    def __init__(
        self,
        input_column: str,
        output_column: str | None = None,  # default: {input}_rewritten
        
        mode: str = "paraphrase",
        # Modes: "paraphrase", "simplify", "formalize", "informalize",
        #        "style", "audience", "length", "elaborate"
        
        preserve: list[str] | None = None,  # aspects to preserve
        num_variations: int = 1,
        
        # Mode-specific
        target_style: str | None = None,
        target_audience: str | None = None,
        target_length: str | None = None,  # "shorter", "longer", "2x", "half"
        
        llm: Model | list[Model] | Sample | None = None,
    ):
        ...

# Usage: Paraphrase for augmentation
>> Rewrite(
    input_column="text",
    mode="paraphrase",
    num_variations=3,
    llm=gpt4,
)

# Usage: Simplify for different audience
>> Rewrite(
    input_column="technical_doc",
    mode="simplify",
    target_audience="high school student",
    llm=gpt4,
)

# Usage: Formalize
>> Rewrite(
    input_column="casual_email",
    mode="formalize",
    llm=gpt4,
)
```

### 13.2 Extract

Pull structured information from unstructured text.

```python
class Extract(Step):
    def __init__(
        self,
        input_column: str,
        
        # What to extract (one of these)
        fields: dict[str, str] | None = None,  # field_name: description
        extractor: str | None = None,          # predefined: "entities", "keywords", etc.
        
        flatten: bool = False,  # flatten extracted dict to columns
        
        llm: Model | list[Model] | Sample | None = None,
    ):
        ...

# Usage: Custom fields
>> Extract(
    input_column="product_description",
    fields={
        "product_name": "The name of the product",
        "price": "Price in dollars (number only)",
        "features": "List of key features",
    },
    flatten=True,
    llm=gpt4,
)
# Output: product_name, price, features columns

# Usage: Named entities
>> Extract(
    input_column="news_article",
    extractor="entities",
    llm=gpt4,
)
# Output: entities = {"persons": [...], "organizations": [...], "locations": [...]}
```

---

## 14. Combination Steps

### 14.1 Pair

Create pairs (or n-tuples) from records within the same dataset.

```python
class Pair(Step):
    def __init__(
        self,
        n: int = 2,                          # pair size (2=pairs, 3=triplets)
        
        strategy: str = "random",
        # Strategies: "random", "sequential", "sliding", "similar", "diverse", "all"
        
        within: str | list[str] | None = None,   # must share these column values
        across: str | list[str] | None = None,   # must differ on these columns
        
        by: str | None = None,                   # for similarity-based strategies
        
        output_format: str = "columns",          # "columns" or "list"
        max_pairs: int | None = None,            # limit total pairs
    ):
        ...

# Usage: Random pairs from same document
>> Pair(n=2, within="document_id")
# Output: chunk_1_*, chunk_2_* columns

# Usage: Similar pairs for multi-hop QA
>> Pair(n=2, strategy="similar", by="text", max_pairs=1000)

# Usage: Triplets with sliding window
>> Pair(n=3, strategy="sliding")

# Usage: Pairs from different categories
>> Pair(n=2, within="topic", across="author")

# Usage: List output format
>> Pair(n=3, output_format="list")
# Output: chunks = [record1, record2, record3], chunk_texts = ["...", "...", "..."]
```

**Output Column Naming:**

With `output_format="columns"` (default):
- For n=2: All columns from record 1 get prefix `chunk_1_`, record 2 gets `chunk_2_`
- For n=3: `chunk_1_`, `chunk_2_`, `chunk_3_`

With `output_format="list"`:
- `chunks` — list of full records
- `{column}_list` — list of values for each original column

### 14.2 Group

Aggregate records by key.

```python
class Group(Step):
    def __init__(
        self,
        by: str | list[str],                    # grouping columns
        
        collect: str | list[str] | None = None, # columns to collect into lists
        output_column: str | None = None,       # name for collected list
        
        agg: dict[str, str] | None = None,      # aggregations: {"new_col": "col:func"}
        # Functions: count, sum, mean, min, max, first, last, collect, concat
        
        min_per_group: int | None = None,       # drop groups smaller than this
        max_per_group: int | None = None,       # limit records per group
    ):
        ...

# Usage: Collect all chunks per document
>> Group(by="document_id", collect="text")
# Output: one record per document with text_list = ["chunk1", "chunk2", ...]

# Usage: With aggregations
>> Group(
    by="product_id",
    collect="review_text",
    agg={
        "avg_rating": "rating:mean",
        "num_reviews": "rating:count",
    },
)
# Output: product_id, review_text_list, avg_rating, num_reviews

# Usage: Filter by group size
>> Group(by="document_id", collect="text", min_per_group=3)
# Only keep documents with at least 3 chunks
```

### 14.3 Concat

Stack multiple datasets vertically.

```python
class Concat(Step):
    def __init__(self, *sources):
        self.sources = sources

# Usage: Combine results from different sources
science = Source.file("science.jsonl") >> LLMStep(...)
history = Source.file("history.jsonl") >> LLMStep(...)

Concat(science, history) >> Sink.jsonl("combined.jsonl")

# Usage: Combine different generation runs
Concat(
    data >> LLMStep(model=gpt4),
    data >> LLMStep(model=claude),
) >> Sink.jsonl("multi_model.jsonl")
```

### 14.4 Join

Merge two datasets horizontally by key.

```python
class Join(Step):
    def __init__(
        self,
        right: "Pipeline | Step",
        on: str | list[str],
        how: str = "inner",  # "inner", "left", "right", "outer"
        suffixes: tuple[str, str] = ("_left", "_right"),
    ):
        ...

# Usage: Merge user data with actions
users = Source.file("users.jsonl")  # {user_id, name}
actions = Source.file("actions.jsonl")  # {user_id, action}

users >> Join(actions, on="user_id") >> Sink.jsonl("enriched.jsonl")
# Output: {user_id, name, action}

# Usage: Merge chosen and rejected responses
chosen >> Join(rejected, on="question_id", suffixes=("_chosen", "_rejected"))
```

### 14.5 Branch and JoinBranches

Split data into parallel processing paths, then merge results.

```python
class Branch(Step):
    def __init__(self, **paths: Step):
        """
        paths: named steps to run in parallel
        Example: Branch(chosen=StepA, rejected=StepB)
        """
        self.paths = paths

class JoinBranches(Step):
    def __init__(
        self,
        on: str | list[str] | None = None,  # join key (auto-generated if None)
        suffixes: dict[str, str] | None = None,  # per-branch suffixes
    ):
        ...

# Usage: Preference data generation
>> Branch(
    chosen=LLMStep(
        prompt="Expert answer: {question}",
        output_columns=["response"],
        model=gpt4,
    ),
    rejected=LLMStep(
        prompt="Brief answer: {question}",
        output_columns=["response"],
        model=gpt35,
    ),
)
>> JoinBranches()
# Output: response_chosen, response_rejected

# Usage: Custom suffixes
>> Branch(
    formal=Rewrite(mode="formalize"),
    casual=Rewrite(mode="informalize"),
)
>> JoinBranches(suffixes={"formal": "_formal", "casual": "_casual"})
```

**Implementation Note:**

Branch internally assigns a unique `_branch_id` to each input record. Each path processes independently. JoinBranches uses `_branch_id` to match records from different paths and merge them.

---

## 15. Pipeline Definition

### 15.1 Pipeline Class

```python
class Pipeline:
    """A directed acyclic graph of steps."""
    
    def __init__(self, steps: list[Step] | None = None):
        self.steps = steps or []
        self._compiled = False
    
    def __rshift__(self, other: Step) -> "Pipeline":
        """Add a step to the pipeline."""
        return Pipeline(self.steps + [other])
    
    def as_step(self, name: str) -> "Pipeline":
        """Name the last step in the pipeline."""
        if self.steps:
            self.steps[-1].as_step(name)
        return self
    
    def compile(self) -> "CompiledPipeline":
        """
        Validate and compile the pipeline.
        - Check that sources and sinks are in correct positions
        - Validate column references
        - Detect branches and joins
        - Assign auto-names to unnamed steps
        """
        ...
    
    def run(
        self,
        # Checkpointing
        checkpoint_dir: str | None = None,
        resume: bool = False,
        resume_from: str | None = None,  # step name
        
        # Execution control
        stop_after: int | str | None = None,  # step number or name
        limit: int | None = None,  # process only first N records
        
        # Batching and parallelism
        batch_size: int = 1,
        max_concurrent: int = 1,
        
        # Rate limiting
        rate_limits: dict[Model, int] | None = None,  # requests per minute per model
        
        # Branch execution
        branch_mode: str = "sequential",  # or "parallel"
        
        # Progress
        show_progress: bool = True,
        log_level: str = "INFO",
    ) -> list[Record] | None:
        """
        Execute the pipeline.
        Returns final records if no Sink, otherwise returns None.
        """
        ...
    
    def run_one(self) -> Record:
        """Process a single record through the pipeline (for testing)."""
        ...
    
    def estimate(self) -> "ExecutionEstimate":
        """Estimate API calls, tokens, and cost before running."""
        ...
    
    def visualize(self) -> str:
        """Return ASCII visualization of the pipeline DAG."""
        ...
```

### 15.2 Pipeline Construction

```python
# Using >> operator
pipeline = (
    Source.huggingface("dataset")
    >> Filter(where={"text": {"$len_gt": 100}})
    >> LLMStep(prompt="...", model=gpt4, ...)
    >> Sink.jsonl("output.jsonl")
)

# With named steps
pipeline = (
    Source.huggingface("dataset")
    .as_step("load")
    
    >> Filter(where={"text": {"$len_gt": 100}})
    .as_step("filter_short")
    
    >> LLMStep(prompt="...", model=gpt4, ...)
    .as_step("generate_questions")
    
    >> Sink.jsonl("output.jsonl")
)

# Multi-line for readability
pipeline = (
    Source.huggingface("dataset")
    >> Sample(n=1000, strategy="diverse", by="text")
    >> Filter(where={"text": {"$len_gt": 200}})
    >> LLMStep(
        prompt="Generate a question: {text}",
        input_columns=["text"],
        output_columns=["question", "answer"],
        model=gpt4,
    )
    >> Score(
        input_columns=["question", "answer"],
        output_column="quality",
        llm=claude,
    )
    >> Filter(where={"quality": {"$gte": 7}})
    >> Sink.hub("username/qa-dataset")
)
```

### 15.3 Pipeline Validation

On `compile()`, the pipeline validates:

1. **Structure:**
   - First step must be a Source (or Seed)
   - Last step should be a Sink (warning if not)
   - No cycles in the DAG

2. **Column references:**
   - `input_columns` in LLMStep exist (from previous steps)
   - `forward_columns` exist
   - `by` columns in Sample/Group/Pair exist

3. **Branch/JoinBranches pairing:**
   - Every Branch has a corresponding JoinBranches
   - Branches don't nest (not supported in v1)

4. **Auto-naming:**
   - Unnamed steps get sequential names: `step_001_source`, `step_002_filter`, etc.

---

## 16. Execution Model

### 16.1 Step-by-Step Execution

The default execution model processes one step at a time across all records:

```
Step 1: Load all records from Source
        → Checkpoint
        
Step 2: Process all records through Step2
        → Checkpoint
        
Step 3: Process all records through Step3
        → Checkpoint
        
...

Step N: Write all records to Sink
        → Complete
```

This enables:
- Checkpointing after every step
- Inspecting intermediate results
- Optimized batching per step
- Per-model rate limiting

### 16.2 Execution Flow

```python
def execute(pipeline, config):
    compiled = pipeline.compile()
    
    # Resume from checkpoint if requested
    start_step = 0
    records = []
    if config.resume and config.checkpoint_dir:
        start_step, records = load_latest_checkpoint(config.checkpoint_dir)
    
    # Execute steps
    for i, step in enumerate(compiled.steps):
        if i < start_step:
            continue  # skip already-completed steps
        
        # Apply limit if testing
        if config.limit and i == 0:
            records = records[:config.limit]
        
        # Process step
        if isinstance(step, Branch):
            records = execute_branch(step, records, config)
        else:
            records = list(step.process(records))
        
        # Checkpoint
        if config.checkpoint_dir:
            save_checkpoint(config.checkpoint_dir, i, step.name, records)
        
        # Progress logging
        log_step_complete(step.name, len(records), elapsed_time)
        
        # Stop if requested
        if config.stop_after and (i == config.stop_after or step.name == config.stop_after):
            break
    
    return records if not isinstance(compiled.steps[-1], Sink) else None
```

### 16.3 LLM Step Execution

Within an LLMStep, execution handles expansion and batching:

```python
def execute_llm_step(step, records, config):
    output_records = []
    
    # Collect all LLM calls to make
    calls = []
    for record in records:
        for prompt in expand(step.prompt, record):
            for model in expand(step.model, record):
                for language in expand(step.language, record):
                    for _ in range(step.num_outputs):
                        calls.append((record, prompt, model, language))
    
    # Group by model for efficient batching
    calls_by_model = group_by(calls, key=lambda c: c[2])
    
    # Execute per model (respecting rate limits)
    for model, model_calls in calls_by_model.items():
        rate_limit = config.rate_limits.get(model)
        
        for batch in batched(model_calls, config.batch_size):
            # Execute batch
            results = execute_batch(model, batch, rate_limit)
            
            # Create output records
            for (record, prompt, model, language), result in zip(batch, results):
                output_record = create_output_record(
                    record, result, step, prompt, model, language
                )
                output_records.append(output_record)
    
    return output_records
```

### 16.4 Branch Execution

```python
def execute_branch(branch_step, records, config):
    # Assign unique IDs for later joining
    for record in records:
        record["_branch_id"] = uuid4().hex
    
    # Execute each branch
    branch_results = {}
    
    if config.branch_mode == "parallel":
        # Run branches in parallel (separate threads/processes)
        with ThreadPoolExecutor() as executor:
            futures = {
                name: executor.submit(execute_step, step, records.copy(), config)
                for name, step in branch_step.paths.items()
            }
            for name, future in futures.items():
                branch_results[name] = future.result()
    else:
        # Run branches sequentially
        for name, step in branch_step.paths.items():
            branch_results[name] = list(step.process(records.copy()))
    
    return branch_results  # JoinBranches will merge these
```

### 16.5 Execution Estimate

Before running, estimate costs:

```python
estimate = pipeline.estimate()
print(estimate)

# Output:
# ExecutionEstimate(
#   total_records=1000,
#   steps=[
#     StepEstimate(name="load", records_out=1000),
#     StepEstimate(name="filter", records_out=~800),
#     StepEstimate(name="generate", records_out=~2400, 
#                  llm_calls=2400, models={"gpt-4": 2400},
#                  estimated_tokens=1_200_000, estimated_cost=36.00),
#     ...
#   ],
#   total_llm_calls=3200,
#   total_estimated_tokens=1_500_000,
#   total_estimated_cost=48.50,
# )
```

---

## 17. Checkpointing and Resumption

### 17.1 Checkpoint Structure

```
checkpoints/
├── manifest.json           # pipeline metadata, current state
├── step_001_load.jsonl     # records after step 1
├── step_002_filter.jsonl   # records after step 2
├── step_003_generate.jsonl # records after step 3 (partial if interrupted)
└── step_003_generate.progress.json  # progress within step 3
```

### 17.2 Manifest File

```json
{
  "pipeline_hash": "abc123...",
  "created_at": "2025-02-01T10:00:00Z",
  "updated_at": "2025-02-01T10:15:00Z",
  "steps": [
    {"index": 0, "name": "load", "status": "complete", "records": 10000},
    {"index": 1, "name": "filter", "status": "complete", "records": 8500},
    {"index": 2, "name": "generate", "status": "in_progress", "records": 3200}
  ],
  "current_step": 2,
  "config": {
    "batch_size": 20,
    "rate_limits": {"gpt-4": 60}
  }
}
```

### 17.3 Progress Within a Step

For long-running LLM steps, track progress within the step:

```json
{
  "step_name": "generate",
  "total_calls": 8500,
  "completed_calls": 3200,
  "last_record_index": 1066,
  "updated_at": "2025-02-01T10:15:00Z"
}
```

### 17.4 Resumption Logic

```python
def resume_from_checkpoint(checkpoint_dir, pipeline):
    manifest = load_manifest(checkpoint_dir)
    
    # Verify pipeline hasn't changed
    if manifest["pipeline_hash"] != pipeline.hash():
        raise PipelineChangedError(
            "Pipeline has changed since checkpoint. Use resume=False to start fresh."
        )
    
    # Find where to resume
    for step_info in manifest["steps"]:
        if step_info["status"] == "complete":
            continue
        elif step_info["status"] == "in_progress":
            # Resume mid-step
            records = load_checkpoint_file(step_info["index"] - 1)
            progress = load_progress_file(step_info["index"])
            return step_info["index"], records, progress
    
    # All steps complete
    return None, None, None
```

### 17.5 Usage

```python
# First run
pipeline.run(checkpoint_dir="./checkpoints")
# Crashes at step 3

# Resume
pipeline.run(checkpoint_dir="./checkpoints", resume=True)
# Continues from step 3

# Resume from specific step (discard later steps)
pipeline.run(checkpoint_dir="./checkpoints", resume_from="generate")

# Run without checkpointing
pipeline.run()
```

---

## 18. Model Abstraction

### 18.1 Model Protocol

```python
from typing import Protocol

class Model(Protocol):
    """Protocol for LLM models."""
    
    @property
    def name(self) -> str:
        """Human-readable model name."""
        ...
    
    @property
    def provider(self) -> str:
        """Provider name (openai, anthropic, etc.)."""
        ...
    
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: type | None = None,  # Pydantic model for structured output
    ) -> str | dict:
        """Generate a completion."""
        ...
```

### 18.2 Provider Implementations

```python
class OpenAIModel(Model):
    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: str | None = None,  # uses OPENAI_API_KEY if None
    ):
        ...

class AnthropicModel(Model):
    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,  # uses ANTHROPIC_API_KEY if None
    ):
        ...

class GeminiModel(Model):
    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ):
        ...

class OllamaModel(Model):
    def __init__(
        self,
        model_id: str = "llama3",
        base_url: str = "http://localhost:11434",
    ):
        ...

# Convenience constructors
gpt4 = OpenAIModel("gpt-4o")
gpt4_mini = OpenAIModel("gpt-4o-mini")
claude = AnthropicModel("claude-sonnet-4-20250514")
claude_haiku = AnthropicModel("claude-haiku-4-20250514")
gemini = GeminiModel("gemini-2.0-flash")
```

### 18.3 Structured Output

LLM steps that expect structured output use Pydantic models:

```python
from pydantic import BaseModel

class QAOutput(BaseModel):
    question: str
    answer: str

# LLMStep internally generates:
response_format = create_model(
    "LLMStepOutput",
    **{col: (str, ...) for col in output_columns}
)
```

The model implementation handles conversion to/from the LLM's native structured output format (JSON mode, tool use, etc.).

---

## 19. Error Handling

### 19.1 Error Types

```python
class DatafastError(Exception):
    """Base class for all Datafast errors."""
    pass

class PipelineValidationError(DatafastError):
    """Pipeline structure is invalid."""
    pass

class ColumnNotFoundError(DatafastError):
    """Referenced column doesn't exist."""
    pass

class LLMError(DatafastError):
    """Error from LLM provider."""
    pass

class CheckpointError(DatafastError):
    """Error with checkpoint loading/saving."""
    pass

class PipelineChangedError(CheckpointError):
    """Pipeline changed since checkpoint was created."""
    pass
```

### 19.2 Error Handling in LLM Steps

```python
class LLMStep:
    def __init__(
        self,
        ...,
        on_error: str = "skip",  # "skip", "raise", "retry"
        max_retries: int = 3,
        retry_delay: float = 1.0,  # seconds
    ):
        ...
```

- `"skip"` — Log the error, skip this record, continue processing
- `"raise"` — Stop the pipeline immediately
- `"retry"` — Retry up to `max_retries` times with exponential backoff

### 19.3 Error Logging

```python
# On error, log with context
logger.error(
    "LLM generation failed",
    step=step.name,
    record_id=record.get("_id"),
    model=model.name,
    prompt_preview=prompt[:100],
    error=str(e),
)
```

### 19.4 Partial Results

When `on_error="skip"`, the pipeline continues and partial results are saved. The final output includes metadata about skipped records:

```python
{
    "records": [...],
    "metadata": {
        "total_input": 1000,
        "total_output": 985,
        "skipped": 15,
        "errors": [
            {"record_id": "abc", "step": "generate", "error": "Rate limit exceeded"},
            ...
        ]
    }
}
```

---

## 20. Module Structure

```
datafast2/
├── __init__.py                 # Public API exports
├── py.typed                    # PEP 561 marker
│
├── core/
│   ├── __init__.py
│   ├── record.py               # Record type definition
│   ├── step.py                 # Step base class, >> operator
│   ├── pipeline.py             # Pipeline class, validation, compilation
│   ├── execution.py            # Execution engine
│   └── checkpoint.py           # Checkpointing logic
│
├── sources/
│   ├── __init__.py
│   ├── base.py                 # Source base class
│   ├── huggingface.py          # Source.huggingface
│   ├── file.py                 # Source.file
│   └── list.py                 # Source.list
│
├── sinks/
│   ├── __init__.py
│   ├── base.py                 # Sink base class
│   ├── jsonl.py                # Sink.jsonl
│   ├── csv.py                  # Sink.csv
│   ├── parquet.py              # Sink.parquet
│   ├── hub.py                  # Sink.hub
│   └── list.py                 # Sink.list
│
├── steps/
│   ├── __init__.py
│   │
│   ├── data/                   # Non-LLM data operations
│   │   ├── __init__.py
│   │   ├── map.py
│   │   ├── flatmap.py
│   │   ├── filter.py
│   │   ├── sample.py
│   │   ├── pair.py
│   │   ├── group.py
│   │   ├── concat.py
│   │   ├── join.py
│   │   └── branch.py
│   │
│   ├── llm/                    # LLM-based steps
│   │   ├── __init__.py
│   │   ├── generate.py         # LLMStep
│   │   ├── classify.py         # Classify
│   │   ├── score.py            # Score
│   │   ├── compare.py          # Compare
│   │   ├── rewrite.py          # Rewrite
│   │   └── extract.py          # Extract
│   │
│   └── __init__.py             # Re-export all steps
│
├── sampling/
│   ├── __init__.py
│   ├── sample.py               # Sample class
│   ├── strategies.py           # Sampling strategy implementations
│   └── embeddings.py           # Embedding-based sampling (diverse)
│
├── seeds/
│   ├── __init__.py
│   └── seed.py                 # Seed builders
│
├── models/
│   ├── __init__.py
│   ├── base.py                 # Model protocol
│   ├── openai.py
│   ├── anthropic.py
│   ├── gemini.py
│   └── ollama.py
│
├── utils/
│   ├── __init__.py
│   ├── logging.py              # Logging configuration
│   ├── progress.py             # Progress tracking
│   ├── validation.py           # Input validation helpers
│   └── cost.py                 # Cost estimation
│
└── presets/                    # Preset pipelines (backward compat, convenience)
    ├── __init__.py
    ├── classification.py
    ├── mcq.py
    ├── preference.py
    └── instruction.py
```

### 20.1 Public API (`__init__.py`)

```python
# Core
from datafast2.core.pipeline import Pipeline
from datafast2.core.record import Record

# Sources and Sinks
from datafast2.sources import Source
from datafast2.sinks import Sink

# Seeds
from datafast2.seeds import Seed

# Sampling
from datafast2.sampling import Sample

# Data Steps
from datafast2.steps.data import (
    Map, FlatMap, Filter, Sample,
    Pair, Group,
    Concat, Join, Branch, JoinBranches,
)

# LLM Steps
from datafast2.steps.llm import (
    LLMStep,
    Classify, Score, Compare,
    Rewrite, Extract,
)

# Models
from datafast2.models import (
    Model,
    OpenAIModel, AnthropicModel, GeminiModel, OllamaModel,
    gpt4, gpt4_mini, claude, claude_haiku, gemini,
)

# Presets
from datafast2.presets import (
    classification_pipeline,
    mcq_pipeline,
    preference_pipeline,
    instruction_pipeline,
)

__version__ = "2.0.0"
```

---

## 21. Implementation Priorities

### Phase 1: Core Foundation (Week 1-2)

**Goal:** Basic pipeline that can chain steps and execute.

1. `Record` type definition
2. `Step` base class with `>>` operator
3. `Pipeline` class with linear execution
4. Basic `Source.list()` and `Sink.list()`
5. `Map`, `Filter` steps
6. Simple execution (no checkpointing yet)

**Test:** `Source.list([...]) >> Map(fn) >> Filter(fn) >> Sink.list()`

### Phase 2: LLM Integration (Week 2-3)

**Goal:** Single LLMStep working with one model.

1. `Model` protocol and `OpenAIModel`
2. `LLMStep` with single prompt, single model
3. Structured output parsing (Pydantic)
4. Basic error handling

**Test:** `Source.list([...]) >> LLMStep(prompt, model) >> Sink.list()`

### Phase 3: Expansion & Sampling (Week 3-4)

**Goal:** Multiple prompts, models, languages; sampling.

1. `Sample` class for data
2. `Sample` integration in LLMStep params
3. List expansion (prompts, models, languages)
4. `num_outputs` multiplication
5. `Seed` builders

**Test:** Multi-model, multi-prompt generation with sampling

### Phase 4: Checkpointing (Week 4-5)

**Goal:** Robust execution with resume capability.

1. Checkpoint saving after each step
2. Manifest file management
3. Resume logic
4. Progress tracking within LLM steps

**Test:** Kill and resume a multi-step pipeline

### Phase 5: Data Combination Steps (Week 5-6)

**Goal:** Pair, Group, Join, Branch.

1. `Pair` with strategies
2. `Group` with aggregations
3. `Concat`
4. `Join`
5. `Branch` and `JoinBranches`

**Test:** Preference dataset with Branch/JoinBranches

### Phase 6: LLM Evaluation Steps (Week 6-7)

**Goal:** Classify, Score, Compare.

1. `Classify`
2. `Score`
3. `Compare`

**Test:** Pipeline with generation → scoring → filtering

### Phase 7: LLM Transformation Steps (Week 7-8)

**Goal:** Rewrite, Extract.

1. `Rewrite`
2. `Extract`

**Test:** Text augmentation pipeline

### Phase 8: Data Sources, Sinks & Remaining Steps (Week 8-9)

**Goal:** FlatMap and remaining file sources/sinks.

1. `FlatMap`
2. `Source.huggingface`, `Source.file`
3. `Sink.jsonl`, `Sink.hub`, etc.

**Test:** Full pipeline from HuggingFace → Hub

### Phase 9: Polish & Presets (Week 9-10)

**Goal:** Quality of life, documentation, backward compatibility.

1. Cost estimation
2. Pipeline visualization
3. Better error messages
4. Preset pipelines
5. Documentation
6. Migration guide from v1

---

## 22. API Quick Reference

### Sources
```python
Source.huggingface(dataset_name, split="train")
Source.file(path)  # .jsonl, .csv, .parquet, .txt
Source.list(records)
Seed.product(Seed.values(...), Seed.expand(...), ...)
```

### Sinks
```python
Sink.jsonl(path)
Sink.csv(path)
Sink.parquet(path)
Sink.hub(repo_id, train_size=0.9)
Sink.list()
```

### Data Operations
```python
Map(fn)
FlatMap(fn)
Filter(fn=..., where=..., expr=...)
Sample(n=..., strategy=..., by=...)
Pair(n, strategy, within, across, by)
Group(by, collect, agg)
Concat(stream1, stream2, ...)
stream1 >> Join(stream2, on=key)
Branch(name1=step1, name2=step2) >> JoinBranches()
```

### LLM Generation
```python
LLMStep(prompt, input_columns, output_columns, model, num_outputs, language)
```

### LLM Evaluation
```python
Classify(labels, input_columns, output_column, llm, multi_label)
Score(input_columns, output_column, range, llm, criteria, rubric)
Compare(column_a, column_b, criteria, llm)
```

### LLM Transformation
```python
Rewrite(input_column, mode, custom_instruction, num_variations, llm)
Extract(input_column, fields, extractor, llm)
```

### Sampling
```python
# As step (data sampling)
>> Sample(n=100)
>> Sample(n=100, strategy="stratified", by="category")
>> Sample(n=100, strategy="diverse", by="text")

# In LLMStep params (config sampling)
LLMStep(prompt=Sample(prompts, n=2), model=Sample(models, n=1, strategy="weighted", by=[...]))
```

### Execution
```python
pipeline.run()
pipeline.run(checkpoint_dir="./ckpt", resume=True)
pipeline.run(limit=10)  # test with 10 records
pipeline.run_one()  # test with 1 record
pipeline.estimate()  # estimate cost
```

---

## Appendix A: Complete Example — Preference Dataset

```python
from datafast2 import (
    Source, Seed, Sample, Filter, Map,
    LLMStep, Score, Branch, JoinBranches,
    Sink, gpt4, gpt4_mini, claude,
)

# Define the pipeline
pipeline = (
    # 1. Load source documents
    Source.huggingface("wikipedia", split="train")
    .as_step("load")
    
    # 2. Sample diverse subset
    >> Sample(n=2000, strategy="diverse", by="text")
    .as_step("sample")
    
    # 3. Filter by length
    >> Filter(where={"text": {"$len_gt": 500, "$len_lt": 5000}})
    .as_step("filter_length")
    
    # 4. Generate questions
    >> LLMStep(
        prompt="Based on this text, generate a thoughtful question that requires understanding the content to answer.\n\nText: {text}",
        input_columns=["text"],
        output_columns=["question"],
        model=gpt4,
        num_outputs=2,
    )
    .as_step("generate_questions")
    
    # 5. Branch for chosen/rejected responses
    >> Branch(
        chosen=LLMStep(
            prompt="""You are an expert assistant. Answer this question thoroughly 
                     and accurately based on the provided context.
                     
                     Context: {text}
                     Question: {question}
                     
                     Provide a comprehensive, well-structured answer.""",
            input_columns=["text", "question"],
            output_columns=["response"],
            model=gpt4,
            temperature=0.7,
        ),
        rejected=LLMStep(
            prompt="""Answer this question based on the context.
                     
                     Context: {text}
                     Question: {question}""",
            input_columns=["text", "question"],
            output_columns=["response"],
            model=gpt4_mini,
            temperature=1.0,
        ),
    )
    .as_step("generate_responses")
    
    >> JoinBranches()
    
    # 6. Score both responses
    >> Score(
        input_columns=["question", "response_chosen"],
        output_column="score_chosen",
        range=(1, 10),
        llm=claude,
        criteria="helpfulness, accuracy, and completeness",
    )
    >> Score(
        input_columns=["question", "response_rejected"],
        output_column="score_rejected",
        range=(1, 10),
        llm=claude,
        criteria="helpfulness, accuracy, and completeness",
    )
    .as_step("score_responses")
    
    # 7. Filter for clear preference margin
    >> Filter(fn=lambda r: r["score_chosen"] >= 7 and r["score_rejected"] <= 5)
    >> Filter(fn=lambda r: r["score_chosen"] - r["score_rejected"] >= 3)
    .as_step("filter_preferences")
    
    # 8. Output
    >> Sink.hub("username/preference-dataset", train_size=0.9, private=False)
)

# Estimate before running
estimate = pipeline.estimate()
print(f"Estimated cost: ${estimate.total_estimated_cost:.2f}")
print(f"Estimated LLM calls: {estimate.total_llm_calls}")

# Run with checkpointing
pipeline.run(
    checkpoint_dir="./preference_checkpoints",
    batch_size=20,
    rate_limits={gpt4: 60, gpt4_mini: 200, claude: 100},
)
```

---

## Appendix B: Complete Example — Multi-Hop QA

> **Note:** This example uses `Chunk`, `Validate`, and `Deduplicate` which are planned as future extensions (see Section 23). The pipeline structure illustrates the intended design; replace those steps with `FlatMap` for chunking and `Filter` + `Score` for validation in the current implementation.

```python
from datafast2 import (
    Source, Sample, Filter, Pair, Group,
    LLMStep, Score,
    Sink, gpt4, claude,
)

pipeline = (
    # 1. Load documents
    Source.huggingface("scientific_papers", split="train")
    .as_step("load")
    
    # 2. Sample papers
    >> Sample(n=500, strategy="diverse", by="abstract")
    .as_step("sample_papers")
    
    # 3. Split into paragraphs via FlatMap (Chunk is a future extension)
    >> FlatMap(lambda r: [
        {**r, "chunk": p.strip()}
        for p in r["full_text"].split("\n\n") if p.strip()
    ])
    .as_step("chunk")
    
    # 4. Filter short chunks
    >> Filter(where={"chunk": {"$len_gt": 200}})
    .as_step("filter_chunks")
    
    # 5. Create pairs of related chunks (same paper)
    >> Pair(
        n=2,
        strategy="similar",
        by="chunk",
        within="paper_id",
        max_pairs=2000,
    )
    .as_step("create_pairs")
    
    # 6. Generate multi-hop questions
    >> LLMStep(
        prompt="""Generate a question that REQUIRES information from BOTH passages to answer.
                 The question should NOT be answerable from either passage alone.
                 
                 Passage 1: {chunk_1_chunk}
                 
                 Passage 2: {chunk_2_chunk}
                 
                 Generate a question and its answer.""",
        input_columns=["chunk_1_chunk", "chunk_2_chunk"],
        output_columns=["question", "answer", "reasoning"],
        model=gpt4,
        num_outputs=2,
    )
    .as_step("generate_qa")
    
    # 7. Score question quality
    >> Score(
        input_columns=["question"],
        output_column="quality",
        range=(1, 5),
        criteria="clarity, specificity, and educational value",
        llm=gpt4,
    )
    >> Filter(where={"quality": {"$gte": 4}})
    .as_step("score_quality")
    
    # 8. Output
    >> Sink.hub("username/multihop-qa-dataset")
)

# Run full pipeline
pipeline.run(
    checkpoint_dir="./multihop_checkpoints",
    batch_size=10,
)
```

---

## Appendix C: Migration from Datafast 1.0

### Classification Dataset

**Before (v1):**
```python
from datafast.datasets import ClassificationDataset
from datafast.schema.config import ClassificationDatasetConfig
from datafast.llms import OpenAIProvider

config = ClassificationDatasetConfig(
    classes=[
        {"name": "positive", "description": "Positive sentiment"},
        {"name": "negative", "description": "Negative sentiment"},
    ],
    num_samples_per_prompt=5,
    output_file="sentiment.jsonl",
    languages={"en": "English"},
)

providers = [OpenAIProvider(model_id="gpt-4")]
dataset = ClassificationDataset(config)
dataset.generate(providers)
```

**After (v2):**
```python
from datafast2 import Seed, LLMStep, Sink, gpt4

pipeline = (
    Seed.product(
        Seed.values("label", ["positive", "negative"]),
        Seed.values("label_description", [
            "Positive sentiment",
            "Negative sentiment",
        ]),
        Seed.values("language", ["English"]),
    )
    >> LLMStep(
        prompt="""Generate 5 diverse text examples in {language} that express 
                 {label} sentiment. {label_description}""",
        input_columns=["label", "label_description", "language"],
        output_columns=["texts"],  # list of 5 texts
        model=gpt4,
    )
    >> FlatMap(lambda r: [
        {"text": t, "label": r["label"], "language": r["language"]}
        for t in r["texts"]
    ])
    >> Sink.jsonl("sentiment.jsonl")
)

pipeline.run()
```

Or use the preset:
```python
from datafast2.presets import classification_pipeline

pipeline = classification_pipeline(
    classes={"positive": "Positive sentiment", "negative": "Negative sentiment"},
    samples_per_class=5,
    languages=["English"],
    model=gpt4,
    output="sentiment.jsonl",
)

pipeline.run()
```

---

## 23. Future Extensions

The following components are intentionally deferred. They follow the same `Step` interface and can be added later without breaking existing pipelines.

**Chunk** — Splits a long text column into smaller, overlapping chunks, yielding one record per chunk. Supports token-based, sentence-based, paragraph-based, and semantic splitting strategies, and is essential for RAG and document-processing pipelines.

**Deduplicate** — Removes duplicate or near-duplicate records by comparing one or more columns. Supports exact matching, fuzzy string similarity, and semantic embedding-based deduplication with a configurable threshold.

**Conversation** — Generates synthetic multi-turn dialogues from a seed topic or message column. Supports configurable turn counts, user and assistant personas, and continuation probability, producing a list of `{role, content}` dicts per record.

**Translate** — Translates a text column into one or more target languages using an LLM, optionally producing a back-translated version for data augmentation. Each target language yields a separate output column.

**Evolve** — Iteratively rewrites content to increase complexity or sophistication over multiple rounds, inspired by the Evol-Instruct technique. Supports strategies such as deepening, broadening, complicating, and adversarial rewriting.

**Refine** — Takes a content column and a corresponding critique column (e.g. from `LLMStep`) and rewrites the content to address the critique, enabling a self-improvement loop.

**Validate** — Performs a boolean LLM-as-judge check against one or more free-text criteria, writing `True`/`False` to an output column. Designed to be chained with `Filter` to keep only records that pass quality gates.

---

*End of Design Document*
