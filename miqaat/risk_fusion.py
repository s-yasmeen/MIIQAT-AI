"""Conservative fusion of deterministic safety rules and AI risk forecasts."""
from dataclasses import dataclass

from .models import RiskResult


@dataclass
class FusedRisk:
    risk: float
    level: str
    rule_level: str
    predicted_risk: float
    prediction_interval: tuple[float, float] | None
    explanation: str


def _level(value):
    if value >= 0.72:
        return "CRITICAL"
    if value >= 0.48:
        return "HIGH"
    if value >= 0.28:
        return "MEDIUM"
    return "LOW"


def fuse_risk(
    rule_result: RiskResult,
    predicted_risk: float,
    prediction_interval=None,
    ai_weight: float = 0.5,
) -> FusedRisk:
    """Blend model risk while preventing the AI estimate from downgrading safety rules.

    If the rule engine reports HIGH or CRITICAL, the fused value cannot fall below
    the rule score. This is a decision-support guardrail, not a certified safety case.
    """
    if not 0.0 <= predicted_risk <= 1.0:
        raise ValueError("predicted_risk must be in [0, 1]")
    if not 0.0 <= ai_weight <= 1.0:
        raise ValueError("ai_weight must be in [0, 1]")
    interval = None
    if prediction_interval is not None:
        if len(prediction_interval) != 2:
            raise ValueError("prediction_interval must contain lower and upper bounds")
        lower, upper = map(float, prediction_interval)
        if lower > upper or lower < 0.0 or upper > 1.0:
            raise ValueError("prediction interval must satisfy 0 <= lower <= upper <= 1")
        interval = (lower, upper)
    blended = (1.0 - ai_weight) * rule_result.total_risk + ai_weight * predicted_risk
    if rule_result.level in ("HIGH", "CRITICAL"):
        fused = max(rule_result.total_risk, blended)
        guardrail = " Safety-rule floor applied; AI forecast cannot downgrade the rule alert."
    else:
        fused = blended
        guardrail = ""
    fused = min(1.0, max(0.0, fused))
    interval_text = (
        f" Predictive interval={interval[0]:.3f}–{interval[1]:.3f}."
        if interval is not None else ""
    )
    return FusedRisk(
        round(fused, 4),
        _level(fused),
        rule_result.level,
        round(predicted_risk, 4),
        interval,
        f"Rule risk={rule_result.total_risk:.3f}; AI forecast risk={predicted_risk:.3f}."
        f"{interval_text}{guardrail}",
    )
