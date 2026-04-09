import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def wrap_env(env: gym.Env, logs_folder: str) -> gym.Env:
    trigger = lambda t: t%100 == 0
    env = RecordVideo(env, 
                      video_folder=f"{logs_folder}/videos/",
                      episode_trigger=trigger
                      )
    return env