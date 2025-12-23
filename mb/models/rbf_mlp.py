import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class RBF(nn.Module):
    num_centers: int
    sigma: float = 1.0
    learnable_centers: bool = True

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        centers = self.param(
            "centers",
            lambda rng: jax.random.uniform(
                rng,
                shape=(self.num_centers, input_dim),
                minval=10.0,
                maxval=50.0,
            )
        )
        if not self.learnable_centers:
            centers = jax.lax.stop_gradient(centers)

        x = x[:, None, :]
        centers = centers[None, :, :]
        dists = jnp.sum((x - centers) ** 2, axis=-1) 
        return jnp.exp(-dists / (2 * self.sigma**2))


class RBFMLP(nn.Module):
    hidden_layers: Sequence[int]
    num_rbf_centers: int = 100
    sigma: float = 5.0

    @nn.compact
    def __call__(self, x):
        x = RBF(num_centers=self.num_rbf_centers, sigma=self.sigma)(x)
        for width in self.hidden_layers:
            x = nn.Dense(width)(x)
            x = nn.softplus(x)
        x = nn.Dense(1)(x)
        return x
