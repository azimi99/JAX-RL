
from gymnasium import spaces

import jax
import jax.numpy as jnp

from flax import nnx

import pytest

from dqn.models.networks import DiscretePolicy, sample_action

def test_policy_network_action_space():
    a = spaces.Discrete(6)
    rngs = nnx.Rngs(0)
    o = spaces.Box(low=-1.0, high=1.0, shape=(12,))
    policy = DiscretePolicy(obs_dim=int(o.shape[-1]), 
                           action_dim=int(a.n), 
                           hidden_dims=(256,256),
                           rngs=rngs)
    logits = policy(o.sample())
    assert logits.shape == (6,)
    
def test_policy_network_action_space():
    
    rngs = nnx.Rngs(0)
    logits = jnp.array([0.1, 0.1, 0.8])
   
    assert logits.shape == (6,)