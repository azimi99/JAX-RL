# environment
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

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
from utils import wrap_env
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
    parser.add_argument('--time_steps', type=int, default=1000_000)
    parser.add_argument('--buffer_size', type=int, default=100_000)
    parser.add_argument('--num_envs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.00)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--start_learning', type=int, default=10_000)
    
    # neural net
    parser.add_argument('--hidden_dim', type=int, default=256)
    
    
    # environment config
    parser.add_argument('--env', type=str, default="CartPole-v1")
    parser.add_argument('--render_mode', type=str, default="rgb_array")
    
    # logging
    parser.add_argument('--logs', type=str, default="./logs")
    
    return parser

def make_env(args, i):
    def thunk():
        env = gym.make(args.env, render_mode=args.render_mode)
        if i == 0:  # only wrap env 0
            env = wrap_env(
                env,
                logs_folder=f'{args.logs}/{args.env}/{args.seed}/videos'
            )
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
    os.makedirs(f"{args.logs}/{args.env}/{args.seed}", exist_ok=True)
    ckpt_dir = ocp.test_utils.erase_and_create_empty(os.path.abspath(f'{args.logs}/{args.env}/{args.seed}/checkpoints'))
    video_dir = ocp.test_utils.erase_and_create_empty(os.path.abspath(f'{args.logs}/{args.env}/{args.seed}/videos'))
    checkpointer = ocp.StandardCheckpointer()
    
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
    optax.clip_by_global_norm(10.0)
    
    lr_schedule = optax.linear_schedule(
        init_value=args.lr,
        end_value=1e-4,
        transition_steps=10000
    )
    tx = optax.adam(learning_rate=lr_schedule)
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
        transition_steps=20_000
    )
    
    batch_size = args.batch_size
    obs, info = env.reset(seed=args.seed)
    done = False
    episode_reward = 0
    
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
        episode_reward += np.mean(reward)
        
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
            if (step // num_envs) % 1000 == 0:
                nnx.update(target_q_net, q_params)
                
                checkpointer.save(ckpt_dir / f'state-{step}', q_params)
            
            if step % (args.num_envs * 100) == 0:
                wandb.log({
                    "train/loss": loss,
                    "env/episode_reward": episode_reward,
                    "env_step": step
                })
    
        obs = next_obs
        if done.any():
            done = False,
            obs, info = env.reset()
            episode_reward = 0 
    wandb.log({"videos": wandb.Video(os.path.join(video_dir, sorted(os.listdir(video_dir))[-1]), caption="final_episode")})
             
    # cleanup
    env.close() 
    wandb.finish()
    
if __name__ == "__main__":
    main()