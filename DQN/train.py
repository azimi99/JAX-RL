# environment
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

# jax imports
import numpy as np
import jax 
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp

# basic imports
import argparse
import types
import logging
import os

# algorithm imports
from utils import save_ckpt_and_video
from dqn.networks.networks import QNetwork
from dqn.networks.buffer \
    import create_replay_buffer,\
        add_transition_batch,\
        sample_batch
from dqn.train.train import train_step

# wandb
import wandb


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "DQN implementation")
    # algorithm
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time_steps', type=int, default=300_000)
    parser.add_argument('--buffer_size', type=int, default=200_000)
    parser.add_argument('--num_envs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.00)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--start_learning', type=int, default=1_000)
    
    # neural net
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    
    # environment config
    parser.add_argument('--env', type=str, default="CartPole-v1")
    parser.add_argument('--render_mode', type=str, default="rgb_array")
    parser.add_argument('--normalize_reward', type=bool, default=False)
    parser.add_argument('--normalize_observation', type=bool, default=False)
    
    # logging
    parser.add_argument('--logs', type=str, default="./logs")
    parser.add_argument('--ckpt_step', type=int, default=10_000)
    
    return parser

def make_env(args, i):
    def thunk():
        env = gym.make(args.env, render_mode=args.render_mode)

        if args.normalize_reward:
            env = NormalizeReward(env)
        if args.normalize_observation:
            env = NormalizeObservation(env)
        return env
    return thunk

@nnx.jit
def sample_action(
    q_net: nnx.Module,
    state: jax.Array,   # (num_envs, state_dim)
    key: jax.Array,
    epsilon: float,
) -> jax.Array:
    q_vals = q_net(state)  # (num_envs, action_dim)

    key_eps, key_act = jax.random.split(key)

    greedy_action = jnp.argmax(q_vals, axis=-1)  # (num_envs,)
    eps_sample = jax.random.uniform(key_eps, shape=(state.shape[0],))
    random_action = jax.random.randint(
        key_act,
        shape=(state.shape[0],),
        minval=0,
        maxval=q_vals.shape[-1],
    )

    return jnp.where(eps_sample > epsilon, greedy_action, random_action)

def main() -> None:
    # Setup training 
    args = args_parser().parse_args()
    ckpt_dir = os.path.abspath(f"{args.logs}/{args.env}/{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(), # Outputs to terminal
            logging.FileHandler(f"{args.logs}/{args.env}/{args.seed}/training.log") # Outputs to file
        ]
    )  
    wandb.init(
        project="rl jax",
        name=f"{args.env}-{args.seed}",
        config=vars(args),   
    ) 
    ## setup environment
    num_timesteps = args.time_steps
    num_envs = args.num_envs
    env = SyncVectorEnv([make_env(args, i) for i in range(num_envs)])
    
    
    ## setup algorithm
    rngs = nnx.Rngs(args.seed)
    
    # q_networks
    q_net = QNetwork(
        obs_dim=env.single_observation_space.shape[0],
        action_dim=env.single_action_space.n,
        hidden_dims=(args.hidden_dim, args.hidden_dim),
        rngs=rngs
    )
    
    target_q_net = QNetwork(
        obs_dim=env.single_observation_space.shape[0],
        action_dim=env.single_action_space.n,
        hidden_dims=(args.hidden_dim, args.hidden_dim),
        rngs=rngs
    )
    
    # Sync target network weights initially
    nnx.update(target_q_net, nnx.state(q_net, nnx.Param))
    
    
    lr_schedule = optax.linear_schedule(
        init_value=args.lr,
        end_value=1e-4,
        transition_steps=10_000
    )
    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adam(learning_rate=lr_schedule)
    )
    optimizer = nnx.Optimizer(q_net, tx=tx)
    buffer = create_replay_buffer(
        capacity=args.buffer_size,
        state_dim=env.single_observation_space.shape,
        action_dim=(1,),  # discrete actions should be scalar       
    )
    logging.info("Allocated replay buffer")
    
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=args.epsilon,
        transition_steps=50_000
    )
    
    batch_size = args.batch_size
    obs, info = env.reset(seed=args.seed)
    done = False
    episode_reward = np.zeros(num_envs, dtype=np.float32)
    loss:float = 0
    for step in range(0, num_timesteps, num_envs):
        key = rngs()
        action = np.array(sample_action(
            q_net=q_net,
            state=obs,
            key=key,
            epsilon=epsilon_schedule(step)
        ))
        
        next_obs, reward, terminated, truncated, info \
            = env.step(action)

        done = terminated | truncated  
        buffer = add_transition_batch(
            buffer=buffer, 
            state=obs,
            action=action,
            next_state=next_obs,
            reward=reward,
            done=terminated,
            num_envs=num_envs
        )
        episode_reward += reward
        
        if buffer.size >= batch_size and step >= args.start_learning:
            key = rngs()
               
            batch = sample_batch(
                buffer=buffer,
                key=key,
                batch_size=batch_size
            )
            
            loss = train_step(
                q_net=q_net,
                target_q_net=target_q_net,
                optimizer=optimizer,
                batch=batch,
                gamma=args.gamma,
                tau=args.tau
            )
            
            q_params = nnx.state(q_net, nnx.Param)
                
            if step % (args.num_envs * args.ckpt_step) == 0:
                
                nnx.update(target_q_net, q_params)
                save_ckpt_and_video(
                    step=step,
                    q_net=q_net,
                    env_id=args.env,
                    ckpt_root=ckpt_dir
                )
    
        obs = next_obs
        if done.any(): # reset for vecenv done automatically
            # log metrics        
            wandb.log({
                "train/loss": float(loss),
                "env/episode_reward": float(np.mean(episode_reward[done])),
                "env_step": int(step)
            })
            episode_reward[done] = 0.0  # reset only that env’s tracker

             
    # cleanup
    env.close() 
    wandb.finish()
    
if __name__ == "__main__":
    main()