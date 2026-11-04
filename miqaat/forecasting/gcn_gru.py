"""Optional graph-aggregation + GRU forecaster (requires PyTorch).

Input x has shape [batch, history, nodes, features]. The adjacency matrix has
shape [nodes, nodes], is row-normalized, and aggregates source-node features
into target nodes. Node ordering must remain fixed across a dataset.
"""
try:
    import torch
    from torch import nn
except ImportError:  # Keep the core simulator usable without ML dependencies.
    torch = None
    nn = None


if nn is not None:
    class GraphTemporalForecaster(nn.Module):
        def __init__(self, input_features: int, nodes: int, hidden_dim: int = 32, horizon: int = 3):
            super().__init__()
            if min(input_features, nodes, hidden_dim, horizon) <= 0:
                raise ValueError("model dimensions must be positive")
            self.nodes = nodes
            self.horizon = horizon
            self.graph_projection = nn.Linear(input_features, hidden_dim)
            self.gru = nn.GRU(nodes * hidden_dim, hidden_dim, batch_first=True)
            self.output = nn.Linear(hidden_dim, horizon * nodes)

        def forward(self, x, adjacency):
            if x.ndim != 4:
                raise ValueError("x must have shape [batch, history, nodes, features]")
            if x.shape[2] != self.nodes:
                raise ValueError("input node dimension does not match model configuration")
            if adjacency.shape != (self.nodes, self.nodes):
                raise ValueError("adjacency must have shape [nodes, nodes]")
            aggregated = torch.einsum("ij,btjf->btif", adjacency, x)
            spatial = torch.relu(self.graph_projection(aggregated))
            temporal = spatial.reshape(x.shape[0], x.shape[1], self.nodes * spatial.shape[-1])
            _, hidden = self.gru(temporal)
            prediction = self.output(hidden[-1])
            return prediction.reshape(x.shape[0], self.horizon, self.nodes)
else:
    class GraphTemporalForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GraphTemporalForecaster requires PyTorch. Install requirements-ml.txt."
            )
