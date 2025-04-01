import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from battery_env import BatteryEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env

consumers_file = "Top_2_Months_Consumers.xlsx"
producers_file = "Top_2_Months_Production.xlsx"

output_folder = "results"
os.makedirs(output_folder, exist_ok=True)

env = BatteryEnv(consumers_file=consumers_file, producers_file=producers_file)
vec_env = make_vec_env(lambda: env, n_envs=1)

model = DDPG("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("ddpg_battery_agent")

test_env = BatteryEnv(consumers_file, producers_file)
obs, _ = test_env.reset()
done = False

P_charge_list = []
P_discharge_list = []
SoC_list = []
Grid_list = []
Waste_list = []
Load_list = []
Prod_list = []
Reward_list = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)

    SoC, P_load, P_prod = obs
    SoC_list.append(SoC)
    Load_list.append(P_load)
    Prod_list.append(P_prod)

    frac_c, frac_d = action
    P_c_max = min(P_prod, (test_env.E_max - SoC) / test_env.eta_c)
    P_d_max = SoC

    P_c = frac_c * P_c_max
    P_d = frac_d * P_d_max
    P_d_real = min(P_d, SoC)

    P_batt = test_env.eta_d * P_d_real
    P_grid = max(0, P_load - P_batt)
    P_waste = max(0, P_prod - P_c)

    Grid_list.append(P_grid)
    Waste_list.append(P_waste)
    P_charge_list.append(P_c)
    P_discharge_list.append(P_d_real)
    Reward_list.append(reward)

T = len(SoC_list)
df = pd.DataFrame({
    "Time": np.arange(T),
    "P_load": Load_list,
    "P_production": Prod_list,
    "P_charge": P_charge_list,
    "P_discharge": P_discharge_list,
    "P_grid": Grid_list,
    "P_waste": Waste_list,
    "Battery_SoC": SoC_list,
    "Reward": Reward_list
})
df["Cost_grid"] = df["P_grid"] * test_env.C_grid
df["Cost_waste"] = df["P_waste"] * test_env.penalty_waste
df["Total_cost"] = df["Cost_grid"] + df["Cost_waste"]

df.to_csv(os.path.join(output_folder, "rl_results.csv"), index=False)

plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["Battery_SoC"], label="Battery SoC (kWh)")
plt.title("Battery State of Charge")
plt.xlabel("Time Step")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Battery_SoC.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["P_charge"], label="Charging (kW)", linestyle='dashed')
plt.plot(df["Time"], df["P_discharge"], label="Discharging (kW)", linestyle='solid')
plt.title("Battery Charging / Discharging")
plt.xlabel("Time Step")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Charge_Discharge.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["P_load"], label="Load", linewidth=2)
plt.plot(df["Time"], df["P_discharge"], label="Battery Discharge", linestyle='dashed')
plt.plot(df["Time"], df["P_grid"], label="Grid", linestyle='dashdot')
plt.plot(df["Time"], df["P_production"], label="Production", linestyle='dotted')
plt.title("Energy Flows Over Time")
plt.xlabel("Time Step")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Energy_Flows.png"))
plt.show()

print(f"\n Total Grid Cost: {df['Cost_grid'].sum():.2f} €")

