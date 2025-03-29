from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from battery_env import BatteryEnv

consumers_file = "Top_2_Months_Consumers.xlsx"
producers_file = "Top_2_Months_Production.xlsx"

env = BatteryEnv(consumers_file=consumers_file, producers_file=producers_file)

# Envolver para paralelismo (opcional)
env = make_vec_env(lambda: env, n_envs=1)

model = DDPG("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/")
model.learn(total_timesteps=100_000)

model.save("ddpg_battery_agent")
