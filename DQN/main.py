# environment
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# jax imports
import numpy as np
import jax 
import jax.numpy as jnp
from flax import nnx
import optax

# basic imports
import argparse
import types
import logging

# algorithm imports
from utils import wrap_env
from dqn.networks.networks import QNetwork
from dqn.networks.buffer \
    import create_replay_buffer, add_transition, sample_batch
from dqn.train.train import train_step

# wandb
import wandb

def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "DQN implementation")
    # seed
    
    # environment config
    parser.add_argument('--env', type=str, default="CartPole-v1")
    parser.add_argument('--render_mode', type=str, default="rgb_array")
    
    # logging
    parser.add_argument('--logs', type=str, default="./logs")
    
    return parser


def sample_action( 
                  q_net: nnx.Module, 
                  state: jax.Array, 
                  key: jax.random.PRNGKey,
                  epsilon: float
                  ) -> int:
    q_vals = q_net(state)
    eps_sample = jax.random.uniform(key=key)
    
    greedy_action = jnp.argmax(q_vals)
    random_action = jax.random.randint(key, shape=(), minval=0, maxval=q_vals.shape[-1])
    
    # Use jnp.where instead of standard if/else
    return int(jnp.where(eps_sample > epsilon, greedy_action, random_action))

def main() -> None:
    # Setup training 
    args = args_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(), # Outputs to terminal
            logging.FileHandler(f"{args.logs}/training.log") # Outputs to file
        ]
    )  
    wandb.init(
        project="rl jax",
        config={
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256  
        },
        
    )
    ## setup environment
    num_timesteps = 500_000
    env = gym.make(args.env, render_mode=args.render_mode)
    env = wrap_env(env, logs_folder=args.logs, num_timesteps=num_timesteps)
    
    ## setup algorithm
    rngs = nnx.Rngs(0)
    
    # q_networks
    q_net = QNetwork(
        obs_dim=env.observation_space.shape[-1],
        action_dim=env.action_space.n,
        hidden_dims=(256,256),
        rngs=rngs
    )
    
    target_q_net = QNetwork(
        obs_dim=env.observation_space.shape[-1],
        action_dim=env.action_space.n,
        hidden_dims=(256,256),
        rngs=rngs
    )
    
    # Sync target network weights initially
    nnx.update(target_q_net, nnx.state(q_net, nnx.Param))
    lr_schedule = optax.linear_schedule(
        init_value=3e-4,
        end_value=1e-4,
        transition_steps=10000
    )
    tx = optax.adam(learning_rate=lr_schedule)
    optimizer = nnx.Optimizer(q_net, tx=tx)
    buffer = create_replay_buffer(
        capacity=100000,
        state_dim=env.observation_space.shape,
        action_dim=(1,),        
    )
    logging.info("Allocated replay buffer")
    
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.05,
        transition_steps=10000
    )
    
    batch_size = 512
    obs, info = env.reset()
    done = False
    episode_reward = 0
    
    for step in range(num_timesteps):
        key = rngs()
        action = sample_action(
            q_net=q_net,
            state=obs,
            key=key,
            epsilon=epsilon_schedule(step)
        )
        
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated | truncated  
        buffer = add_transition(
            buffer=buffer, 
            state=obs,
            action=action,
            next_state=next_obs,
            reward=reward,
            done=terminated
        )
        episode_reward += reward
        if buffer.size >= batch_size:
            # start updates after
            key = rngs()
            batch = sample_batch(
                buffer=buffer,
                key=key,
                batch_size=batch_size,
            )
            loss = train_step(
                q_net=q_net,
                target_q_net=target_q_net,
                optimizer=optimizer,
                batch=batch,
                gamma=0.99,
                tau=0.001
            )
            if step % 10_000 == 0:  
                wandb.log({
                    "train/loss": loss,
                    "env/episode_reward": episode_reward,
                    "step": step
                })
            
        obs = next_obs
        if done:
            done = False,
            obs, info = env.reset()
            episode_reward = 0 
             
    # cleanup
    env.close() 
    
if __name__ == "__main__":
    main()