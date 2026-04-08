from dqn.networks.buffer import ReplayBuffer, Batch, create_replay_buffer, add_transition, sample_batch

import jax
import jax.numpy as jnp

import gymnasium
import pytest


@pytest.fixture
def buffer_setup():
   obs_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(3,)) 
   action_space = gymnasium.spaces.Discrete(3)
   buffer = create_replay_buffer(
       capacity=100,
       state_dim=obs_space.shape,
       action_dim=(action_space.n,)
   )
   yield (buffer, obs_space, action_space)
   
    

def test_create_replay_buffer(buffer_setup):
   buffer, obs_space, action_space = buffer_setup
   assert buffer.capacity == 100
   assert buffer.states.shape == (100, *obs_space.shape)
   assert buffer.actions.shape == (100, action_space.n)

def test_add_transition(buffer_setup):
    buffer, obs_space, action_space = buffer_setup
    write_idx = buffer.write_idx
    buffer = add_transition(
        buffer, 
        state=obs_space.sample(),
        action=action_space.sample(),
        next_state=obs_space.sample(),
        reward=1.0,
        done=0.0
    )
    assert buffer.rewards[write_idx] == 1.0
    assert buffer.write_idx > write_idx

def test_sample_batch(buffer_setup):
    buffer, obs_space, action_space = buffer_setup
    
    batch = sample_batch(
        buffer=buffer,
        key=jax.random.PRNGKey(0),
        batch_size=10
    )
    
    assert batch.states.shape == (10, *obs_space.shape)
    