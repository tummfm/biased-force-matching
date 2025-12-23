from flax import linen as nn
from typing import Sequence

class MLP(nn.Module):
    hidden_layers: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for width in self.hidden_layers:
            x = nn.Dense(width)(x)
            x = nn.softplus(x)
        x = nn.Dense(1)(x)
        return x