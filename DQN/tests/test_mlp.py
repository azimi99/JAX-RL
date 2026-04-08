import jax
import jax.numpy as jnp

from flax import nnx
from dqn.networks.mlp import MLP

def test_mlp_out_shape():
    x = jnp.ones((12,))
    model = MLP(in_dim=12, hidden_dims=(256,256), out_dim=4,rngs=nnx.Rngs(0))
    o = model(x)
    assert o.shape == (4,)