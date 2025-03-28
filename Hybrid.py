import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pyswarm import pso
from scipy.optimize import minimize
import time

#%% Configuração inicial
E_max = np.array([2500])
eta_c = np.array([0.95])
eta_d = np.array([0.95])
initial_SoC = E_max * 0.5
C_grid = 0.1
AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15
output_folder = "hybrid_pso"
os.makedirs(output_folder, exist_ok=True)

#%% Carregar dados
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = consumers_data.shape[0]
P_load_matrix = consumers_data.values.T
P_production_matrix = producers_data.values.T
n_consumers = P_load_matrix.shape[0]
n_producers = P_production_matrix.shape[0]

P_disch_max_flat = np.tile(E_max, T)
P_grid_max = np.tile(P_load_matrix.max(axis=1), T)

#%% Função para denormalizar

def denormalize_vector(x, lower, upper):
    return lower + x * (upper - lower)

#%% Função Objetivo

def objective(x):
    penalty_weights = {
        "max_power": 200,
        "SoC_violation": 200,
        "over_discharge": 200,
        "fallback_grid": 1e12,
        "grid_priority_violation": 10,
        "unused_production": 100
    }

    offset1 = n_producers * T
    offset2 = offset1 + T
    offset3 = offset2 + n_consumers * T

    P_c = denormalize_vector(x[:offset1], 0, P_production_matrix.flatten()).reshape(n_producers, T)
    P_d = denormalize_vector(x[offset1:offset2], 0, P_disch_max_flat).reshape(1, T)
    P_grid = denormalize_vector(x[offset2:offset3], 0, P_grid_max).reshape(n_consumers, T)

    SoC = np.zeros((1, T + 1))
    SoC[0, 0] = initial_SoC[0]
    penalty = 0.0

    SoC_min = 0.10 * E_max[0]
    P_c_max = 0.25 * E_max[0]
    P_d_max = 0.25 * E_max[0]

    for t in range(T):
        P_c_total = np.sum(P_c[:, t])

        if P_c_total > P_c_max:
            penalty += penalty_weights["max_power"] * (P_c_total - P_c_max)
            scale = P_c_max / P_c_total if P_c_total > 0 else 0
            P_c[:, t] *= scale

        if P_d[0, t] > P_d_max:
            penalty += penalty_weights["max_power"] * (P_d[0, t] - P_d_max)
            P_d[0, t] = P_d_max

        if P_d[0, t] > SoC[0, t]:
            penalty += penalty_weights["over_discharge"] * (P_d[0, t] - SoC[0, t])
            P_d[0, t] = SoC[0, t]

        SoC[0, t+1] = SoC[0, t] + eta_c[0] * P_c_total - P_d[0, t]

        if SoC[0, t+1] < SoC_min:
            penalty += penalty_weights["SoC_violation"] * (SoC_min - SoC[0, t+1])
            SoC[0, t+1] = SoC_min
        elif SoC[0, t+1] > E_max[0]:
            penalty += penalty_weights["SoC_violation"] * (SoC[0, t+1] - E_max[0])
            SoC[0, t+1] = E_max[0]

        for p in range(n_producers):
            if P_c[p, t] > P_production_matrix[p, t]:
                penalty += penalty_weights["max_power"] * (P_c[p, t] - P_production_matrix[p, t])

        unused_production = np.sum(P_production_matrix[:, t]) - P_c_total
        if unused_production > 0:
            penalty += penalty_weights["unused_production"] * unused_production

        battery_energy = P_d[0, t] * eta_d[0]
        grid_energy = np.sum(P_grid[:, t])
        total_demand = np.sum(P_load_matrix[:, t])
        supplied_energy = battery_energy + grid_energy

        if supplied_energy < total_demand:
            penalty += penalty_weights["fallback_grid"] * (total_demand - supplied_energy)

        battery_energy_available = SoC[0, t] * eta_d[0]
        if battery_energy_available >= total_demand and grid_energy > 0:
            penalty += penalty_weights["grid_priority_violation"] * grid_energy

    global last_grid_energy_sum
    last_grid_energy_sum = np.sum(P_grid) * C_grid
    return last_grid_energy_sum + penalty

#%% Normalizar vetor vindo do Simplex
simplex_df = pd.read_csv("simplex_plots/simplex_results.csv")
P_charge_simplex = simplex_df["P_charge"].values
P_discharge_simplex = simplex_df["P_discharge"].values
P_grid_simplex = simplex_df["P_grid"].values

x_seed_real = np.concatenate([
    np.tile(P_charge_simplex, n_producers),
    P_discharge_simplex,
    np.tile(P_grid_simplex, n_consumers)
])

x_upper = np.concatenate([
    P_production_matrix.flatten(),
    P_disch_max_flat,
    P_grid_max
])

x_seed_norm = np.clip(x_seed_real / x_upper, 0, 1)

#%% Executar PSO com partícula semente
n_vars = len(x_seed_norm)

last_grid_energy_sum = 0.0
x_best, fopt = pso(
    objective,
    lb=np.zeros(n_vars),
    ub=np.ones(n_vars),
    swarmsize=50,
    maxiter=300,
    minstep=1e-6,
    debug=True
)

#%% Refinamento local com scipy.optimize.minimize
print("\n🔧 Starting local optimization...")
start = time.time()
res = minimize(
    objective,
    x_best,
    method="SLSQP",
    bounds=[(0, 1)] * n_vars,
    options={"maxiter": 500, "disp": True, "iprint": 2}
)
print("✅ Local optimization completed.")
print(f"⏱️ Duration: {time.time() - start:.2f} seconds")

x_refined = res.x
fopt_refined = res.fun

#%% Processar resultados refinados
P_c_opt = denormalize_vector(x_refined[:n_producers*T], 0, P_production_matrix.flatten()).reshape(n_producers, T)
P_d_opt = denormalize_vector(x_refined[n_producers*T:n_producers*T+T], 0, P_disch_max_flat).reshape(1, T)
P_grid_opt = denormalize_vector(x_refined[n_producers*T+T:], 0, P_grid_max).reshape(n_consumers, T)

SoC = np.zeros((1, T + 1))
SoC[0, 0] = initial_SoC[0]
P_d_real = np.zeros_like(P_d_opt)

for t in range(T):
    P_d_real[0, t] = min(P_d_opt[0, t], SoC[0, t])
    SoC[0, t+1] = SoC[0, t] + eta_c[0] * np.sum(P_c_opt[:, t]) - P_d_real[0, t]
    SoC[0, t+1] = np.clip(SoC[0, t+1], 0, E_max[0])

P_charge_total = np.sum(P_c_opt, axis=0)
P_discharge_total = P_d_real[0]
Battery_SoC_total = SoC[0, 1:]
P_grid_total = np.sum(P_grid_opt, axis=0)

results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_grid": P_grid_total,
    "P_charge": P_charge_total,
    "P_discharge": P_discharge_total,
    "Battery_SoC": Battery_SoC_total
})
results_df["Cost"] = results_df["P_grid"] * C_grid
results_df.to_csv(os.path.join(output_folder, "hybrid_refined_results.csv"), index=False)

#%% Plotting Results

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="Battery State of Charge (kWh)", linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery SoC Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_charge"], label="Battery Charging (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharging (kW)", linestyle='solid', linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Power (kW)")
plt.title("Charging and Discharging Patterns")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Charging_Discharging_Patterns.png"))
plt.show()

#%% Resultados finais
print(f"\n🔍 Simplex Cost (no penalties): {np.sum(P_grid_simplex) * C_grid:.2f} €")
print(f"✅ Hybrid PSO Total Cost (with penalties): {fopt:.2f} €")
print(f"💡 Hybrid PSO Grid Cost Only (no penalties): {last_grid_energy_sum:.2f} €")
print(f"🔧 Refined Cost After Local Optimization: {fopt_refined:.2f} €")

#%% Gráfico: Consumo, Descarga da Bateria, Importação da Rede e Produção
total_consumo = np.sum(P_load_matrix, axis=0)
total_producao = np.sum(P_production_matrix, axis=0)

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], total_consumo, label="Total Consumption (kW)", linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharge (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Import (kW)", linestyle='dashdot', linewidth=2)
plt.plot(results_df["Time"], total_producao, label="Total Production (kW)", linestyle='dotted', linewidth=2)

plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Power (kW)")
plt.title("Energy Flows Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Energy_Flows_Over_Time.png"))
plt.show()

