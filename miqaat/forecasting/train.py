"""Optional PyTorch training/evaluation helpers for graph-temporal forecasting.

The caller is responsible for chronological, scenario/video-separated data splits
and training-only normalization. This module does not create or claim a dataset.
"""
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int


def _require_torch():
    if torch is None:
        raise ImportError("Training requires PyTorch; install requirements-ml.txt.")


def train_model(
    model,
    train_x,
    train_y,
    adjacency,
    validation_data=None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 12,
    seed: int = 2026,
):
    """Train with early stopping; inputs must already be normalized from train data."""
    _require_torch()
    if epochs <= 0 or learning_rate <= 0 or patience <= 0:
        raise ValueError("epochs, learning_rate and patience must be positive")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    a = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    if x.ndim != 4 or y.ndim != 3:
        raise ValueError("train_x must be [batch,time,nodes,features]; train_y [batch,horizon,nodes]")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError("training inputs and targets have incompatible batch/node dimensions")

    validation = None
    if validation_data is not None:
        val_x, val_y = validation_data
        val_x = torch.as_tensor(val_x, dtype=torch.float32, device=device)
        val_y = torch.as_tensor(val_y, dtype=torch.float32, device=device)
        if val_x.ndim != 4 or val_y.ndim != 3 or val_x.shape[0] != val_y.shape[0]:
            raise ValueError("validation tensors have invalid dimensions")
        validation = (val_x, val_y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_losses, validation_losses = [], []
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(x, a)
        if prediction.shape != y.shape:
            raise ValueError("model output shape must match target shape [batch,horizon,nodes]")
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        train_value = float(loss.detach().cpu())
        train_losses.append(train_value)

        model.eval()
        with torch.no_grad():
            if validation is None:
                monitored = train_value
            else:
                val_prediction = model(validation[0], a)
                monitored = float(criterion(val_prediction, validation[1]).cpu())
        validation_losses.append(monitored)

        if monitored < best_loss:
            best_loss, best_epoch, stale = monitored, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingHistory(train_losses, validation_losses, best_epoch)


def regression_metrics(prediction, target):
    """Return aggregate MAE/RMSE; report per-horizon/per-zone metrics separately."""
    _require_torch()
    pred = torch.as_tensor(prediction, dtype=torch.float32)
    truth = torch.as_tensor(target, dtype=torch.float32)
    if pred.shape != truth.shape or pred.numel() == 0:
        raise ValueError("prediction and target must have identical non-empty shapes")
    error = pred - truth
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(torch.sqrt((error ** 2).mean())),
    }
