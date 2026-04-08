import jax
import jax.numpy as jnp

from flax import nnx
from typing import Any, Callable, List


class MLP(nnx.Module):
    def __init__(self, 
                in_dim: int,
                hidden_dims: tuple[int,...],
                out_dim: int, 
                rngs: nnx.Rngs, 
                activation_fn: Callable = nnx.relu):
        dims = (in_dim, *hidden_dims, out_dim)
        self.activation_fn = activation_fn
        self.layers = [nnx.Linear(dims[i], dims[i+1], rngs=rngs) for i in range(len(dims) - 1)]
        
    def __call__(self, x) -> jnp.ndarray:
        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x

if __name__ == "__main__":
    x = jnp.ones(12)
    model = MLP(3, x.shape[0], 32, rngs=nnx.Rngs(0))
    out = model(x)
    assert out.shape == (32,)
    nnx.display(model)
        
        
