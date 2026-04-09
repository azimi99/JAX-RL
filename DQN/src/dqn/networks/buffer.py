import jax
import jax.numpy as jnp
from flax import struct

import numpy as np

from typing import Any, Callable, List
from dataclasses import dataclass


@struct.dataclass
class ReplayBuffer:
    states: jax.Array
    actions: jax.Array
    next_states: jax.Array
    rewards: jax.Array
    dones: jax.Array
    write_idx: jax.Array
    capacity: int = struct.field(pytree_node=False) # metadata
    
@struct.dataclass
class Batch:
    states: jax.Array
    actions: jax.Array
    next_states: jax.Array
    rewards: jax.Array
    dones: jax.Array 

def create_replay_buffer(
    capacity: int,
    state_dim: tuple[int, ...],
    action_dim: tuple[int,...],  
) -> ReplayBuffer:
    return ReplayBuffer(
        states=jnp.zeros(shape=(capacity, *state_dim)),
        actions=jnp.zeros(shape=(capacity, *action_dim)),
        next_states=jnp.zeros(shape=(capacity, *state_dim)),
        rewards=jnp.zeros(shape=(capacity,)),
        dones=jnp.zeros(shape=(capacity,)),
        capacity=capacity,
        write_idx=0,
    )

def add_transition(buffer: ReplayBuffer, 
                   state: jax.Array,
                   action: jax.Array,
                   next_state: jax.Array,
                   reward: jax.Array,
                   done: jax.Array) -> ReplayBuffer:
    idx = buffer.write_idx % buffer.capacity
    
    return ReplayBuffer(
        states=buffer.states.at[idx].set(state),
        actions=buffer.actions.at[idx].set(action),
        next_states=buffer.next_states.at[idx].set(next_state),
        rewards=buffer.rewards.at[idx].set(reward),
        dones=buffer.dones.at[idx].set(done),
        write_idx=buffer.write_idx + 1,
        capacity=buffer.capacity,
    )
    
def sample_batch(
    buffer: ReplayBuffer,
    key: jax.Array,
    batch_size: int
) -> Batch:
    idx = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=batch_size)
    return Batch(
        states=buffer.states[idx],
        actions=buffer.actions[idx],
        next_states=buffer.next_states[idx],
        rewards=buffer.rewards[idx],
        dones=buffer.dones[idx]
    )
