# jax imports
import jax
import jax.numpy as jnp
# flax imports
from flax import nnx
from typing import Any, Callable, List
# mlp backbone
from dqn.networks.mlp import MLP

from gymnasium import spaces

from typeguard import typechecked

class QNetwork(nnx.Module):
    @typechecked
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dims: tuple[int,...],
                 rngs: nnx.Rngs):
        self.mlp_backbone = MLP(
            in_dim=obs_dim,
            out_dim=action_dim,
            hidden_dims=hidden_dims,
            rngs=rngs
        )
    def __call__(self, obs: jax.Array) -> jax.Array:
        
        return self.mlp_backbone(obs)



if __name__ == "__main__":
    a = spaces.Discrete(6)
    rngs = nnx.Rngs(0)
    o = spaces.Box(low=-1.0, high=1.0, shape=(12,))
    policy = QNetwork(obs_dim=int(o.shape[-1]), 
                           action_dim=int(a.n), 
                           hidden_dims=(256,256),
                           rngs=rngs)
    print(policy(o.sample()))
    