from __future__ import annotations

from typing import Any


class BaseSelector:
    """Interface for AutoTSP selector strategies."""

    def predict(self, features: dict[str, Any], remaining_budget: float):
        raise NotImplementedError


__all__ = ["BaseSelector"]
