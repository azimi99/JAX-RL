import os
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import jax
import jax.numpy as jnp
from flax import nnx

import orbax.checkpoint as ocp


def save_ckpt_and_video(
    *,
    step: int,
    q_net: nnx.Module,
    env_id: str,
    ckpt_root: str,
):
    step_dir = Path(ckpt_root) / f"step_{step:09d}"
    video_dir = step_dir / "videos"
    ckpt_dir = step_dir / "ckpt"
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- 1) Save checkpoint (params) ---
    params = nnx.state(q_net, nnx.Param)  # PyTree
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(ckpt_dir), params, force=True)

    # --- 2) Record one eval episode ---
    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = RecordVideo(
        eval_env,
        video_folder=str(video_dir),
        episode_trigger=lambda ep: True,  # record the next (and only) episode we run
        name_prefix=f"eval_{step:09d}",
    )

    obs, info = eval_env.reset()
    done = False
    while not done:
        # greedy action for eval video
        q = q_net(jnp.asarray(obs)[None, ...])          # (1, A)
        action = int(jnp.argmax(q, axis=-1)[0].item())  # python int for Gym

        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = bool(terminated or truncated)

    eval_env.close()

