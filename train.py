from stable_baselines3 import PPO
from env.car_env import get_env
import configs
import os

os.makedirs(configs.LOG_DIR, exist_ok=True)
os.makedirs(configs.MODEL_DIR, exist_ok=True)

env = get_env()

model_path = os.path.join(configs.MODEL_DIR, configs.MODEL_NAME + ".zip")

if os.path.exists(model_path):
    print("Loading existing model...")
    model = PPO.load(model_path, env)
else:
    print("Creating new model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=configs.LEARNING_RATE,
        n_steps=configs.N_STEPS,
        batch_size=configs.BATCH_SIZE,
        n_epochs=configs.N_EPOCHS,
        gamma=configs.GAMMA,
        tensorboard_log=configs.LOG_DIR
    )

model.learn(total_timesteps=configs.TOTAL_TIMESTEPS)
model.save(model_path)
