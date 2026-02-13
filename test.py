from stable_baselines3 import PPO
from env.car_env import get_env
import configs
import time

env = get_env(render_mode="human")

model = PPO.load(f"{configs.MODEL_DIR}/{configs.MODEL_NAME}", env)

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    time.sleep(0.01)
