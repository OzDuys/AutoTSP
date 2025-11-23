from AutoTSP.selectors.base import BaseSelector
from AutoTSP.selectors.selector_random_forest import RandomForestSelector
from AutoTSP.selectors.selector_rule_based import RuleBasedSelector


def get_selector(name: str = "rule_based", **kwargs) -> BaseSelector:
    if name == "rule_based":
        return RuleBasedSelector()
    if name in {"random_forest", "rf"}:
        return RandomForestSelector(**kwargs)
    raise ValueError(f"Unknown selector: {name}")


__all__ = ["BaseSelector", "RuleBasedSelector", "RandomForestSelector", "get_selector"]
