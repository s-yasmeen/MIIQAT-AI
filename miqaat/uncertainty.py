"""Split-conformal intervals for calibrated forecast uncertainty."""


def conformal_radius(calibration_residuals, alpha: float = 0.10) -> float:
    """Return the finite-sample absolute-residual conformal radius.

    Residuals must come from a held-out calibration set that is separate from
    model fitting and final test data.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    residuals = sorted(abs(float(value)) for value in calibration_residuals)
    if not residuals:
        raise ValueError("calibration_residuals cannot be empty")
    rank = min(len(residuals), int(__import__("math").ceil((len(residuals) + 1) * (1.0 - alpha))))
    return residuals[rank - 1]


def prediction_interval(prediction: float, radius: float):
    if radius < 0:
        raise ValueError("radius cannot be negative")
    return (float(prediction) - radius, float(prediction) + radius)
