import jax
import jax.numpy as jnp
import optax
from flax import nnx

from dqn.networks.buffer import Batch




@nnx.jit
def loss_fn(q_net: nnx.Module,
            target_q_net: nnx.Module, 
            batch: Batch,
            gamma: float):
    states = batch.states # (B, S)
    actions = batch.actions # (B, 1), action is a single integer
    #Expand dimensions to prevent (B,) and (B, 1) broadcasting to (B, B)
    dones = jnp.expand_dims(batch.dones, axis=1)       # (B,) -> (B, 1)
    rewards = jnp.expand_dims(batch.rewards, axis=1)   # (B,) -> (B, 1)
    next_states = batch.next_states
    state_action_values = jnp.take_along_axis(nnx.vmap(q_net)(states), actions.astype(int), axis=1) # (B, 1)
    next_state_values = jnp.max(nnx.vmap(target_q_net)(next_states), axis=1, keepdims=True) # (B, 1)
    target_val = rewards + gamma * (1 - dones) * next_state_values
    td_error = target_val - state_action_values
    
    
    # huber loss
    loss = jnp.where(jnp.abs(td_error) <= 1.0, 
                     (0.5 * td_error ** 2), 
                     jnp.abs(td_error) - 0.5)
    
    return jnp.mean(loss)
    
    

@nnx.jit
def train_step(q_net: nnx.Module,
               target_q_net:nnx.Module,
               optimizer: nnx.Optimizer,
               batch: Batch,
               gamma: float,
               tau: float) -> jax.Array:
    loss, grads = nnx.value_and_grad(loss_fn, argnums=0, has_aux=False)(q_net, 
                                              target_q_net, 
                                              batch, 
                                              gamma)
    
    grads = jax.tree.map(lambda g: jnp.clip(g, -1000, 1000), grads)
    optimizer.update(grads)
    
    # soft update of target network
    
    q_params = nnx.state(q_net, nnx.Param)
    target_q_params = nnx.state(target_q_net, nnx.Param)
    
    new_target_params = jax.tree.map(
        lambda t_theta, theta: tau*theta + (1-tau)*t_theta,
        target_q_params,
        q_params,

    )
    nnx.update(target_q_net, new_target_params)
    
    return loss


    
