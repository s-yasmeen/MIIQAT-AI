"""Dependency-free forecasting baselines for per-zone time series."""


def persistence_forecast(values, horizons=(1, 3, 6)):
    if not values:
        raise ValueError("values cannot be empty")
    return {int(h): float(values[-1]) for h in horizons}


def moving_average_forecast(values, window=3, horizons=(1, 3, 6)):
    if window <= 0 or not values:
        raise ValueError("window must be positive and values cannot be empty")
    estimate = sum(float(value) for value in values[-window:]) / min(window, len(values))
    return {int(h): estimate for h in horizons}


def linear_trend_forecast(values, horizons=(1, 3, 6)):
    """Least-squares linear extrapolation over equally spaced observations."""
    if not values:
        raise ValueError("values cannot be empty")
    y = [float(value) for value in values]
    n = len(y)
    if n == 1:
        return {int(h): y[0] for h in horizons}
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    denominator = sum((x - x_mean) ** 2 for x in range(n))
    slope = sum((x - x_mean) * (value - y_mean) for x, value in enumerate(y)) / denominator
    intercept = y_mean - slope * x_mean
    return {int(h): intercept + slope * (n - 1 + int(h)) for h in horizons}
