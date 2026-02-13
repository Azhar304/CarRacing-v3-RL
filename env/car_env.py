import gymnasium as gym
import cv2
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

class PreprocessCarRacing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # Add channel dimension
        obs = np.expand_dims(obs, axis=-1)
        return obs

def make_env(render_mode="rgb_array"):
    def _init():
        # CHANGE HERE: v3 instead of v2
        env = gym.make("CarRacing-v3", render_mode=render_mode)
        env = PreprocessCarRacing(env)
        return env
    return _init

def get_env(render_mode="rgb_array"):
    env = DummyVecEnv([make_env(render_mode)])
    env = VecFrameStack(env, n_stack=4)
    return env
