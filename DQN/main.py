# environment
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# jax imports
import numpy as np
import jax 
from jax import numpy as jnp

# basic imports
import argparse
import types



def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog = "DQN implementation")
    parser.add_argument('--test_parser', type=bool, default=True)
    return parser

def main() -> None:
    args = args_parser().parse_args()
    print(args.test_parser)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    trigger = lambda t: t%10 == 0
    env = RecordVideo(env, 
                      video_folder="./logs/videos/",
                      episode_trigger=trigger
                      )
    num_episodes = 100
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
        
    
    
    

if __name__ == "__main__":
    main()