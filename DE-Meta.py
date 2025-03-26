import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import os

#%% Parâmetros fixos
N_batteries = 3
E_max = np.array([300, 500, 200])
eta_c = np.array([0.9, 0.95, 0.92])
eta_d = np.array([0.9, 0.95, 0.92])
battery_degradation_factor = np.array([0.0003, 0.0005, 0.0004])
initial_SoC = E_max * 0.5
C_grid = 0.1

AGGREGATION_INTERVAL = 240
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15

output_folder = "de_plots"
os.makedirs(output_folder, exist_ok=True)

#%% Carregar dados
consumers_data = pd.read_excel("Dataset_Consumers_2.xlsx")
producers_data = pd.read_excel("Dataset_Producers_2.xlsx")

num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = consumers_data.shape[0]
P_load = consumers_data.sum(axis=1).values
P_production = producers_data.sum(axis=1).values

#%% Metaheurístic Vars
P_prod_max = np.tile(P_production, (N_batteries, 1))
P_prod_max_flat = P_prod_max.flatten()
P_disch_max_flat = np.tile(E_max, T)
P_grid_max = max(P_load) * 1.5

#%% Função de denormalização vetorizada
def denormalize_vector(x, lower, upper):
    return lower + x * (upper - lower)

#%% Função Objetivo
def objective(x):
    P_c = denormalize_vector(x[:N_batteries * T], 0, P_prod_max_flat).reshape(N_batteries, T)
    P_d = denormalize_vector(x[N_batteries * T:2 * N_batteries * T], 0, P_disch_max_flat).reshape(N_batteries, T)
    P_grid = denormalize_vector(x[2 * N_batteries * T:], 0, P_grid_max)

    SoC = np.zeros((N_batteries, T + 1))
    SoC[:, 0] = initial_SoC
    penalty = 0

    for t in range(T):
        for i in range(N_batteries):
            available_energy = SoC[i, t]
            if available_energy <= 0:
                if P_d[i, t] > 0:
                    penalty += 1e6 * P_d[i, t]
                P_d[i, t] = 0
            elif P_d[i, t] > available_energy:
                penalty += 1e6 * (P_d[i, t] - available_energy)
                P_d[i, t] = available_energy

        SoC[:, t+1] = SoC[:, t] + eta_c * P_c[:, t] - P_d[:, t]

        for i in range(N_batteries):
            if SoC[i, t+1] < 0 or SoC[i, t+1] > E_max[i]:
                penalty += 1e6 * abs(SoC[i, t+1] - np.clip(SoC[i, t+1], 0, E_max[i]))
                SoC[i, t+1] = np.clip(SoC[i, t+1], 0, E_max[i])

        P_ren = P_production[t]
        P_in = np.sum(P_d[:, t] * eta_d) + P_grid[t]
        P_out = np.sum(P_c[:, t] / eta_c) + P_load[t]

        if P_in + P_ren < P_out - 1e-3:
            penalty += 1e6 * (P_out - (P_in + P_ren))

        if np.sum(P_c[:, t]) > P_ren:
            penalty += 1e6 * (np.sum(P_c[:, t]) - P_ren)

    grid_cost = np.sum(P_grid) * C_grid
    degradation_cost = np.sum([battery_degradation_factor[i] * np.sum(P_d[i]) for i in range(N_batteries)])
    total_cost = grid_cost + degradation_cost + penalty
    return total_cost

#%% Otimização
bounds = [(0, 1)] * (2 * N_batteries * T + T)

result = differential_evolution(
    objective,
    bounds=bounds,
    strategy='randtobest1bin',
    mutation=(0.5, 1),
    recombination=0.7,
    popsize=15,
    maxiter=100,
    polish=True,
    disp=True,
    updating="deferred"
)


#%% Resultados finais
x_opt = result.x
P_c_opt = denormalize_vector(x_opt[:N_batteries * T], 0, P_prod_max_flat).reshape(N_batteries, T)
P_d_opt = denormalize_vector(x_opt[N_batteries * T:2 * N_batteries * T], 0, P_disch_max_flat).reshape(N_batteries, T)
P_grid_opt = denormalize_vector(x_opt[2 * N_batteries * T:], 0, P_grid_max)

SoC = np.zeros((N_batteries, T + 1))
SoC[:, 0] = initial_SoC
P_d_real = np.zeros_like(P_d_opt)

for t in range(T):
    for i in range(N_batteries):
        P_d_real[i, t] = min(P_d_opt[i, t], SoC[i, t])
        SoC[i, t+1] = SoC[i, t] + eta_c[i] * P_c_opt[i, t] - P_d_real[i, t]
        SoC[i, t+1] = np.clip(SoC[i, t+1], 0, E_max[i])

P_charge_total = np.sum(P_c_opt, axis=0)
P_discharge_total = np.sum(P_d_real, axis=0)
Battery_SoC_total = np.sum(SoC[:, 1:], axis=0)

results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_grid": P_grid_opt,
    "P_charge": P_charge_total,
    "P_discharge": P_discharge_total,
    "Battery_SoC": Battery_SoC_total
})
results_df["Cost"] = results_df["P_grid"] * C_grid
results_df.to_csv(os.path.join(output_folder, "de_results.csv"), index=False)

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Cost"], label="Cost Over Time (€)", linewidth=2, color='r')
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Cost (€)")
plt.title("Cost Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Cost_Over_Time.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="Battery State of Charge (SoC) (kWh)", linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery SoC Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Power Import (kW)", linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Grid Power Import (kW)")
plt.title("Grid Power Import Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Grid_Power_Import.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_charge"], label="Battery Charging (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharging (kW)", linestyle='solid', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Power (kW)")
plt.title("Charging and Discharging Patterns")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Charging_Discharging_Patterns.png"))
plt.show()

print(f"Custo Total (DE): {results_df['Cost'].sum():.2f} €")