"""Transparent additive attributions; this module does not claim to implement SHAP."""


def additive_attributions(features, coefficients, baseline=None):
    """Decompose a linear score into signed feature contributions.

    The sum of contributions equals the score difference from the chosen baseline.
    """
    if set(features) != set(coefficients):
        raise ValueError("features and coefficients must have identical keys")
    baseline = baseline or {key: 0.0 for key in features}
    if set(baseline) != set(features):
        raise ValueError("baseline must contain the same feature keys")
    contributions = {
        key: (float(features[key]) - float(baseline[key])) * float(coefficients[key])
        for key in features
    }
    return sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)


def explain_additive_score(features, coefficients, baseline=None, top_k=5):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    contributions = additive_attributions(features, coefficients, baseline)
    selected = contributions[:top_k]
    delta = sum(value for _, value in contributions)
    return {
        "method": "linear_additive_contribution",
        "score_delta": delta,
        "top_contributors": [
            {"feature": name, "contribution": value} for name, value in selected
        ],
        "limitation": "Model-specific additive decomposition; not SHAP or causal explanation.",
    }
