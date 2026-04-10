import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def wrap_env(env: gym.Env, logs_folder: str) -> gym.Env:
    env = RecordVideo(env, 
                      video_folder=f"{logs_folder}",
                      episode_trigger=lambda ep:ep%10 == 0
                      )
    return env

