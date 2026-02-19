"""Sample class for selecting items from collections using various strategies."""

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

from datafast_v2.core.step import Step
from datafast_v2.core.types import Record


class Sample(Step):
    """
    Select items from a collection using various strategies.

    Can be used both as:
    - A pipeline step to sample data records
    - A configuration object to sample prompts, models, etc. in LLMStep

    Strategies:
        - "uniform": Equal probability random selection (default)
        - "first": First N items
        - "last": Last N items
        - "systematic": Every Nth item (uses `step` parameter)
        - "top": Highest values by column/function
        - "bottom": Lowest values by column/function
        - "weighted": Probability proportional to weights
        - "stratified": Maintain category distribution
        - "gaussian": Weight by normal distribution around center

    Examples:
        >>> # 100 random records
        >>> Sample(n=100)

        >>> # First 50 records
        >>> Sample(n=50, strategy="first")

        >>> # Top 100 by score
        >>> Sample(n=100, strategy="top", by="score")

        >>> # Stratified by category
        >>> Sample(n=500, strategy="stratified", by="category")

        >>> # Sample from config items
        >>> Sample(["prompt1", "prompt2", "prompt3"], n=2)
    """

    VALID_STRATEGIES = frozenset([
        "uniform",
        "first",
        "last",
        "systematic",
        "top",
        "bottom",
        "weighted",
        "stratified",
        "gaussian",
    ])

    def __init__(
        self,
        items: list | Iterable | None = None,
        *,
        n: int | None = None,
        frac: float | None = None,
        strategy: str = "uniform",
        by: str | Callable | list[float] | None = None,
        ascending: bool = False,
        center: float | None = None,
        std: float | None = None,
        step: int | None = None,
        seed: int | None = None,
        replace: bool = False,
    ) -> None:
        """
        Initialize a Sample.

        Args:
            items: Items to sample from. If None, operates on pipeline records.
            n: Absolute count of items to select.
            frac: Fraction of items to select (0.0 to 1.0).
            strategy: Selection strategy. One of: uniform, first, last,
                systematic, top, bottom, weighted, stratified, gaussian.
            by: Column name, callable, or list of weights for strategy.
            ascending: For top/bottom, whether to sort ascending.
            center: Center value for gaussian strategy.
            std: Standard deviation for gaussian strategy.
            step: Step size for systematic strategy.
            seed: Random seed for reproducibility.
            replace: Whether to sample with replacement.
        """
        super().__init__()

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {strategy}. "
                f"Valid strategies: {sorted(self.VALID_STRATEGIES)}"
            )

        if n is not None and frac is not None:
            raise ValueError("Cannot specify both 'n' and 'frac'")

        if strategy == "systematic" and step is None:
            raise ValueError("'step' parameter required for systematic strategy")

        if strategy == "gaussian" and (center is None or std is None):
            raise ValueError(
                "'center' and 'std' parameters required for gaussian strategy"
            )

        if strategy in ("top", "bottom", "weighted", "stratified", "gaussian"):
            if by is None:
                raise ValueError(f"'by' parameter required for {strategy} strategy")

        self._items: list | None = list(items) if items is not None else None
        self._n = n
        self._frac = frac
        self._strategy = strategy
        self._by = by
        self._ascending = ascending
        self._center = center
        self._std = std
        self._step = step
        self._seed = seed
        self._replace = replace
        self._rng: random.Random | None = None

    def _get_rng(self) -> random.Random:
        """Get or create the random number generator."""
        if self._rng is None:
            self._rng = random.Random(self._seed)
        return self._rng

    def _compute_n(self, total: int) -> int:
        """Compute the number of items to select."""
        if self._n is not None:
            return min(self._n, total) if not self._replace else self._n
        if self._frac is not None:
            return max(1, int(total * self._frac))
        return total

    def _get_value(self, item: Any, by: str | Callable | None) -> Any:
        """Extract value from item using 'by' parameter."""
        if by is None:
            return item
        if callable(by):
            return by(item)
        if isinstance(item, dict):
            return item.get(by)
        return getattr(item, by, None)

    def _sample_uniform(self, items: list, n: int) -> list:
        """Uniform random sampling."""
        rng = self._get_rng()
        if self._replace:
            return [rng.choice(items) for _ in range(n)]
        return rng.sample(items, min(n, len(items)))

    def _sample_first(self, items: list, n: int) -> list:
        """Take first N items."""
        return items[:n]

    def _sample_last(self, items: list, n: int) -> list:
        """Take last N items."""
        return items[-n:] if n <= len(items) else items

    def _sample_systematic(self, items: list, n: int) -> list:
        """Take every Nth item."""
        step = self._step or 1
        result = items[::step]
        return result[:n] if n < len(result) else result

    def _sample_top(self, items: list, n: int) -> list:
        """Take top N items by value."""
        sorted_items = sorted(
            items,
            key=lambda x: self._get_value(x, self._by),
            reverse=not self._ascending,
        )
        return sorted_items[:n]

    def _sample_bottom(self, items: list, n: int) -> list:
        """Take bottom N items by value (same as top with ascending=True)."""
        sorted_items = sorted(
            items,
            key=lambda x: self._get_value(x, self._by),
            reverse=False,
        )
        return sorted_items[:n]

    def _sample_weighted(self, items: list, n: int) -> list:
        """Weighted random sampling."""
        rng = self._get_rng()

        if isinstance(self._by, list):
            weights = self._by
            if len(weights) != len(items):
                raise ValueError(
                    f"Weight list length ({len(weights)}) must match "
                    f"items length ({len(items)})"
                )
        else:
            weights = [self._get_value(item, self._by) for item in items]

        total_weight = sum(weights)
        if total_weight == 0:
            return self._sample_uniform(items, n)

        normalized = [w / total_weight for w in weights]

        if self._replace:
            result = []
            for _ in range(n):
                r = rng.random()
                cumulative = 0.0
                for item, prob in zip(items, normalized):
                    cumulative += prob
                    if r <= cumulative:
                        result.append(item)
                        break
            return result
        else:
            selected = []
            available_items = list(items)
            available_weights = list(normalized)

            for _ in range(min(n, len(items))):
                total = sum(available_weights)
                if total == 0:
                    break
                renormalized = [w / total for w in available_weights]

                r = rng.random()
                cumulative = 0.0
                for i, (item, prob) in enumerate(zip(available_items, renormalized)):
                    cumulative += prob
                    if r <= cumulative:
                        selected.append(item)
                        available_items.pop(i)
                        available_weights.pop(i)
                        break

            return selected

    def _sample_stratified(self, items: list, n: int) -> list:
        """Stratified sampling maintaining category distribution."""
        rng = self._get_rng()

        groups: dict[Any, list] = defaultdict(list)
        for item in items:
            key = self._get_value(item, self._by)
            groups[key].append(item)

        total = len(items)
        result = []

        for key, group_items in groups.items():
            group_frac = len(group_items) / total
            group_n = max(1, int(n * group_frac))
            group_n = min(group_n, len(group_items))

            if self._replace:
                result.extend([rng.choice(group_items) for _ in range(group_n)])
            else:
                result.extend(rng.sample(group_items, group_n))

        if len(result) > n:
            result = rng.sample(result, n)

        return result

    def _sample_gaussian(self, items: list, n: int) -> list:
        """Sample with gaussian weighting around center."""
        rng = self._get_rng()

        center = self._center
        std = self._std

        weights = []
        for item in items:
            value = self._get_value(item, self._by)
            if value is None:
                weights.append(0.0)
            else:
                weight = math.exp(-0.5 * ((value - center) / std) ** 2)
                weights.append(weight)

        total_weight = sum(weights)
        if total_weight == 0:
            return self._sample_uniform(items, n)

        normalized = [w / total_weight for w in weights]

        if self._replace:
            result = []
            for _ in range(n):
                r = rng.random()
                cumulative = 0.0
                for item, prob in zip(items, normalized):
                    cumulative += prob
                    if r <= cumulative:
                        result.append(item)
                        break
            return result
        else:
            selected = []
            available_items = list(items)
            available_weights = list(normalized)

            for _ in range(min(n, len(items))):
                total = sum(available_weights)
                if total == 0:
                    break
                renormalized = [w / total for w in available_weights]

                r = rng.random()
                cumulative = 0.0
                for i, (item, prob) in enumerate(zip(available_items, renormalized)):
                    cumulative += prob
                    if r <= cumulative:
                        selected.append(item)
                        available_items.pop(i)
                        available_weights.pop(i)
                        break

            return selected

    def _apply_strategy(self, items: list, n: int) -> list:
        """Apply the sampling strategy to items."""
        strategy_map = {
            "uniform": self._sample_uniform,
            "first": self._sample_first,
            "last": self._sample_last,
            "systematic": self._sample_systematic,
            "top": self._sample_top,
            "bottom": self._sample_bottom,
            "weighted": self._sample_weighted,
            "stratified": self._sample_stratified,
            "gaussian": self._sample_gaussian,
        }

        return strategy_map[self._strategy](items, n)

    def pick(self, n: int | None = None) -> list:
        """
        Materialize the sample immediately and return a plain list.

        Useful when you want to sample once upfront, then use the
        selected items exhaustively.

        Args:
            n: Override the number of items to select.

        Returns:
            A plain list of selected items.

        Example:
            >>> selected = Sample(prompts, n=5, seed=42).pick()
            >>> # selected is now a plain list of 5 prompts
        """
        if self._items is None:
            raise ValueError("Cannot call pick() on a Sample without items")

        effective_n = n if n is not None else self._compute_n(len(self._items))
        return self._apply_strategy(self._items, effective_n)

    def sample(self) -> list:
        """
        Perform sampling and return selected items.

        Returns:
            List of selected items.
        """
        return self.pick()

    def process(self, records: Iterable[Record]) -> Iterable[Record]:
        """
        Process records as a pipeline step.

        When used as a step, Sample collects all input records,
        applies the sampling strategy, and yields selected records.
        """
        all_records = list(records)

        if not all_records:
            return

        n = self._compute_n(len(all_records))
        logger.info(
            f"Sampling {n} records from {len(all_records)} "
            f"using strategy '{self._strategy}'"
        )

        selected = self._apply_strategy(all_records, n)
        yield from selected

    def __iter__(self):
        """Allow iteration over all items (for exhaustive use)."""
        if self._items is None:
            raise ValueError("Cannot iterate over a Sample without items")
        return iter(self._items)

    def __len__(self) -> int:
        """Return number of items."""
        if self._items is None:
            raise ValueError("Cannot get length of a Sample without items")
        return len(self._items)

    @property
    def items(self) -> list:
        """Return the items list."""
        if self._items is None:
            raise ValueError("Sample has no items (used as pipeline step)")
        return self._items
