import jax
import jax.numpy as jnp
from flax import nnx
import optax

from dqn.networks.networks import QNetwork
from dqn.train.train import loss_fn, train_step
from dqn.networks.buffer import Batch

import pytest

@pytest.fixture
def network_setup():
    rngs = nnx.Rngs(0)
    q1 = QNetwork(
        3,
        3,
        (10,10),
        rngs=rngs
    )
    q2 = QNetwork(
        3,
        3,
        (10,10),
        rngs=rngs
    )
    
    batch = Batch(
        states=jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
            ],
            dtype=jnp.float32,
        ),
        actions=jnp.array([0, 1, 2, 1], dtype=jnp.int32).reshape(-1,1),
        rewards=jnp.array([1.0, 0.5, -0.2, 0.0], dtype=jnp.float32),
        next_states=jnp.array(
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0],
                [1.1, 1.2, 1.3],
            ],
            dtype=jnp.float32,
        ),
        dones=jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    yield (q1, q2, batch)


def test_loss_fn(network_setup):

    q1, q2, batch = network_setup
    loss = loss_fn(
        q_net=q1,
        target_q_net=q2,
        batch=batch,
        gamma=0.98    
    )
    
    assert isinstance(loss, jax.Array)
    assert not jnp.isnan(loss).any()
    
def test_train_step(network_setup):
    q1, q2, batch = network_setup
    optimizer = nnx.Optimizer(q1, optax.adam(1e-3))
    loss = train_step(
        q_net=q1,
        target_q_net=q2,
        optimizer=optimizer,
        batch=batch,
        gamma=0.98,
        tau=0.1
    )
    assert isinstance(loss, jax.Array)
    