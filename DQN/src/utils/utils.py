import gymnasium as gym
from gymnasium.wrappers import RecordVideo

def wrap_env(env: gym.Env, logs_folder: str, num_timesteps: int) -> gym.Env:
    trigger = lambda t: 0 == t % 50_000
    env = RecordVideo(env, 
                      video_folder=f"{logs_folder}/videos/",
                      step_trigger=trigger
                      )
    return env