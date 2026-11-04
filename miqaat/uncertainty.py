"""Split-conformal intervals for calibrated forecast uncertainty."""
import math


def conformal_radius(calibration_residuals, alpha: float = 0.10) -> float:
    """Return the finite-sample absolute-residual split-conformal radius.

    Residuals must come from a held-out calibration set separate from model fitting
    and final test data. Returns infinity when the requested finite-sample rank is
    n+1; this is the conservative finite-sample result, not the largest observed
    residual.
    """
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and between 0 and 1")
    residuals = [abs(float(value)) for value in calibration_residuals]
    if not residuals:
        raise ValueError("calibration_residuals cannot be empty")
    if any(not math.isfinite(value) for value in residuals):
        raise ValueError("calibration residuals must all be finite")
    residuals.sort()
    rank = math.ceil((len(residuals) + 1) * (1.0 - alpha))
    if rank > len(residuals):
        return math.inf
    return residuals[rank - 1]


def prediction_interval(prediction: float, radius: float):
    prediction = float(prediction)
    radius = float(radius)
    if not math.isfinite(prediction):
        raise ValueError("prediction must be finite")
    if math.isnan(radius) or radius < 0:
        raise ValueError("radius must be non-negative and not NaN")
    return prediction - radius, prediction + radius
