from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.car_env import get_env
import configs

env = get_env()

model = PPO.load(f"{configs.MODEL_DIR}/{configs.MODEL_NAME}", env)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward} +/- {std_reward}")
