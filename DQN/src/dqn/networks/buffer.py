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
    size: jax.Array
    priorities: jax.Array
    capacity: int = struct.field(pytree_node=False) # metadata
    
@struct.dataclass
class Batch:
    states: jax.Array
    actions: jax.Array
    next_states: jax.Array
    rewards: jax.Array
    dones: jax.Array
    probs: jax.Array
    indices: jax.Array
    weights: jax.Array

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
        priorities =jnp.ones(shape=(capacity,)),
        write_idx=0,
        size=0
    )

def add_transition(buffer: ReplayBuffer, 
                   state: jax.Array,
                   action: jax.Array,
                   next_state: jax.Array,
                   reward: jax.Array,
                   done: jax.Array) -> ReplayBuffer:
    idx = buffer.write_idx % buffer.capacity
    max_priority = jnp.max(buffer.priorities[:buffer.size])
    priority = jnp.where(buffer.size > 0, max_priority, 1.0)
    
    return ReplayBuffer(
        states=buffer.states.at[idx].set(state),
        actions=buffer.actions.at[idx].set(action),
        next_states=buffer.next_states.at[idx].set(next_state),
        rewards=buffer.rewards.at[idx].set(reward),
        dones=buffer.dones.at[idx].set(done),
        write_idx=buffer.write_idx + 1,
        size=jnp.minimum(buffer.size + 1, buffer.capacity),
        capacity=buffer.capacity,
        priorities= buffer.priorities.at[idx].set(priority)
        
    )

def add_transition_batch(buffer: ReplayBuffer, 
                   state: jax.Array,# (num_envs, *state_dim)
                   action: jax.Array,
                   next_state: jax.Array,
                   reward: jax.Array,
                   done: jax.Array,
                   num_envs: int) -> ReplayBuffer:
    # idx = buffer.write_idx % buffer.capacity
    indices = (buffer.write_idx + jnp.arange(num_envs)) % buffer.capacity
    max_priority = jax.lax.cond(
        buffer.size > 0,
        lambda _: jnp.max(buffer.priorities[indices]),
        lambda _: 1.0,
        operand=None,
    )
    priority = jnp.where(buffer.size > 0, max_priority, 1.0)
    return ReplayBuffer(
        states=buffer.states.at[indices].set(state),
        actions=buffer.actions.at[indices].set(action[:, None]),
        next_states=buffer.next_states.at[indices].set(next_state),
        rewards=buffer.rewards.at[indices].set(reward),
        dones=buffer.dones.at[indices].set(done),
        write_idx=buffer.write_idx + num_envs,
        size=jnp.minimum(buffer.size + num_envs, buffer.capacity),
        capacity=buffer.capacity,
        priorities= buffer.priorities.at[indices].set(priority)
    )
    
def sample_batch(
    buffer: ReplayBuffer,
    key: jax.Array,
    batch_size: int,
    alpha: float
) -> Batch:

    valid_priorities = buffer.priorities[:buffer.size]
    scaled = valid_priorities ** alpha
    probs = scaled / jnp.sum(scaled)
    indices = jax.random.choice(
        key,
        a=buffer.size,
        shape=(batch_size,),
        replace=True,
        p=probs,
    )
    return Batch(
        states=buffer.states[indices],
        actions=buffer.actions[indices],
        next_states=buffer.next_states[indices],
        rewards=buffer.rewards[indices],
        dones=buffer.dones[indices],
        weights=is_weights_fn(probs[indices], buffer.size, beta=0.5),
        probs=probs[indices],
        indices=indices
    )

def is_weights_fn(probs, size, beta:float):
    weights = (1.0 / (size * probs)) ** beta
    weights = weights / jnp.max(weights)
    return weights


def update_priorities(buffer: ReplayBuffer, indices, td_errors, eps: float = 1e-6):
    new_priorities = jnp.abs(td_errors) + eps
    return buffer.replace(
        priorities=buffer.priorities.at[indices].set(jnp.squeeze(new_priorities))
    )