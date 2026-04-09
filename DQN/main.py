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


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "DQN implementation")
    # seed
    
    # environment config
    parser.add_argument('--env', type=str, default="CartPole-v1")
    parser.add_argument('--render_mode', type=str, default="rgb_array")
    
    # logging
    parser.add_argument('--logs', type=str, default="./logs")
    
    return parser

def sample_action(env: gym.Env, 
                  q_net: nnx.Module, 
                  state: jax.Array, 
                  key: jax.random.PRNGKey,
                  epsilon: float
                  ) -> int:
    q_vals = q_net(state)
    eps_sample = jax.random.uniform(key=key)
    if eps_sample > epsilon:
        return int(jnp.argmax(q_vals))
    else:
        return env.action_space.sample()

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
    # setup environment
    env = gym.make(args.env, render_mode=args.render_mode)
    env = wrap_env(env, logs_folder=args.logs)
    # setup algorithm
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
        action_dim=int(env.action_space.n),
        hidden_dims=(256,256),
        rngs=rngs
    )
    optimizer = nnx.Optimizer(q_net, tx=optax.adam(1e-3))
    buffer = create_replay_buffer(
        capacity=1000,
        state_dim=env.observation_space.shape,
        action_dim=(env.action_space.n,),        
    )
    logging.info("Allocated replay buffer")
    num_episodes = 1000
    step = 0
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            key = rngs()
            action = sample_action(
                env,
                q_net=q_net,
                state=obs,
                key=key,
                epsilon=0.05
            )
            next_obs, reward, terminated, truncated, info = env.step(action)
            step+=1
            done = terminated 
            buffer = add_transition(
                buffer=buffer, 
                state=obs,
                action=action,
                next_state=next_obs,
                reward=reward,
                done=done
            )
            if step >= 100:
                # start updates after
                key = rngs()
                batch = sample_batch(
                    buffer=buffer,
                    key=key,
                    batch_size=32,
                )
                loss = train_step(
                    q_net=q_net,
                    target_q_net=target_q_net,
                    optimizer=optimizer,
                    batch=batch,
                    gamma=0.99,
                    tau=0.001
                )  
   
            obs = next_obs    
    # cleanup
    env.close()
        
    
    
    

if __name__ == "__main__":
    main()