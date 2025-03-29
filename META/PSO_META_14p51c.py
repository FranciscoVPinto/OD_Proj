# %% PSO Optimization with Multi-Agent Energy Modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pyswarm import pso

# Parameters
E_max = 2500
eta_c = 0.95
eta_d = 0.95
initial_SoC = E_max * 0.5
C_grid = 0.1
lambda_waste = 0.2
AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15
output_folder = "pso_multiagent_plots"
os.makedirs(output_folder, exist_ok=True)

# Load and aggregate data
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)
num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = len(consumers_data)
N_consumers = consumers_data.shape[1]
N_producers = producers_data.shape[1]

P_load_matrix = consumers_data.values
P_prod_matrix = producers_data.values

# Lower and upper bounds
lb = np.zeros(T * N_producers + T)
ub = [P_prod_matrix[t, j] for t in range(T) for j in range(N_producers)] + [E_max] * T
ub = np.array(ub)

# Objective function
def objective(x):
    P_charge_matrix = x[:T * N_producers].reshape((T, N_producers))
    P_discharge = x[T * N_producers:]
    P_charge_total = P_charge_matrix.sum(axis=1)

    SoC = np.zeros(T + 1)
    SoC[0] = initial_SoC
    grid_cost, waste_cost, penalty = 0.0, 0.0, 0.0

    for t in range(T):
        real_P_d = min(P_discharge[t], SoC[t])
        SoC[t+1] = np.clip(SoC[t] + eta_c * P_charge_total[t] - real_P_d, 0, E_max)
        P_available = eta_d * real_P_d
        grid = max(0.0, P_load_matrix[t].sum() - P_available)
        waste = np.sum(np.maximum(0.0, P_prod_matrix[t] - P_charge_matrix[t]))

        grid_cost += grid * C_grid
        waste_cost += waste * lambda_waste

        if P_discharge[t] > 0 and P_charge_total[t] > 0:
            penalty += 1e4 * min(P_discharge[t], P_charge_total[t])

    penalty += 1e4 * abs(SoC[-1] - initial_SoC)

    energia_carregada_util = np.sum(P_charge_total) * eta_c
    energia_total_descarga = np.sum([min(P_discharge[t], SoC[t]) for t in range(T)])
    delta_soc = SoC[-1] - SoC[0]
    energia_total_disponivel = energia_carregada_util + delta_soc

    if energia_total_descarga - energia_total_disponivel > 1e-3:
        return 1e12

    return grid_cost + waste_cost + penalty

# Run PSO
x_opt, fopt = pso(objective, lb, ub, swarmsize=250, maxiter=150, minstep=1e-6, debug=True)

# Extract results
P_charge_matrix_opt = x_opt[:T * N_producers].reshape((T, N_producers))
P_discharge_opt = x_opt[T * N_producers:]
P_charge_total = P_charge_matrix_opt.sum(axis=1)

SoC = np.zeros(T + 1)
SoC[0] = initial_SoC
P_grid = np.zeros(T)
P_waste = np.zeros(T)

for t in range(T):
    real_P_d = min(P_discharge_opt[t], SoC[t])
    SoC[t+1] = np.clip(SoC[t] + eta_c * P_charge_total[t] - real_P_d, 0, E_max)
    P_grid[t] = max(0.0, P_load_matrix[t].sum() - eta_d * real_P_d)
    P_waste[t] = np.sum(np.maximum(0.0, P_prod_matrix[t] - P_charge_matrix_opt[t]))

results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_charge_total": P_charge_total,
    "P_discharge": P_discharge_opt,
    "Battery_SoC": SoC[1:],
    "P_grid": P_grid,
    "P_waste": P_waste,
    "Cost_grid": P_grid * C_grid,
    "Energy_waste": P_waste,
})

results_df.to_csv(os.path.join(output_folder, "PSO_MultiAgent_Results.csv"), index=False)

# Plots
plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="SoC (kWh)")
plt.title("Battery State of Charge Over Time")
plt.xlabel("8h Time Blocks")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Battery_SoC.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["P_charge_total"], label="Charge (kW)")
plt.plot(results_df["Time"], results_df["P_discharge"], label="Discharge (kW)")
plt.title("Charging / Discharging")
plt.xlabel("8h Time Blocks")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Charge_Discharge.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], P_load_matrix.sum(axis=1), label="Load", linewidth=2)
plt.plot(results_df["Time"], P_discharge_opt, label="Battery Discharge", linestyle='--')
plt.plot(results_df["Time"], P_grid, label="Grid Import", linestyle='-.')
plt.plot(results_df["Time"], P_prod_matrix.sum(axis=1), label="Production", linestyle=':')
plt.title("Energy Flow Over Time")
plt.xlabel("8h Time Blocks")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Energy_Flow.png"))
plt.show()

# Summary
energia_carregada_util = np.sum(P_charge_total) * eta_c
delta_soc = SoC[-1] - SoC[0]
energia_total_disponivel = energia_carregada_util + delta_soc
SoC_temp = np.zeros(T + 1)
SoC_temp[0] = initial_SoC
real_P_d_list = []
for t in range(T):
    real_P_d = min(P_discharge_opt[t], SoC_temp[t])
    SoC_temp[t+1] = np.clip(SoC_temp[t] + eta_c * P_charge_total[t] - real_P_d, 0, E_max)
    real_P_d_list.append(real_P_d)
energia_total_descarga = np.sum(real_P_d_list)

print("\n===== CHECK DE CONSISTÊNCIA ENERGÉTICA DA BATERIA =====")
print(f"Energia carregada (útil): {energia_carregada_util:.2f} kWh")
print(f"Variação de SoC:          {delta_soc:.2f} kWh")
print(f"Total disponível:         {energia_total_disponivel:.2f} kWh")
print(f"Total descarregada:       {energia_total_descarga:.2f} kWh")

if energia_total_descarga - energia_total_disponivel > 1e-3:
    print("⚠️ ERRO: A bateria forneceu mais do que podia!")
else:
    print("✅ OK: A bateria está energeticamente consistente.")

print(f"\n💡 Grid cost: {results_df['Cost_grid'].sum():.2f} €")
print(f"🗑️ Total energy wasted: {results_df['Energy_waste'].sum():.2f} kWh")
