from __future__ import annotations

from AutoTSP.selectors.selector_rule_based import RuleBasedSelector


class Selector:
    """Compatibility wrapper pointing to the rule-based selector."""

    def predict(self, features: dict, remaining_budget: float) -> type:
        return RuleBasedSelector().predict(features, remaining_budget)


__all__ = ["Selector"]
