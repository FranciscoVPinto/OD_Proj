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

AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15

output_folder = "meta_plots"
os.makedirs(output_folder, exist_ok=True)

#%% Carregar dados
consumers_data = pd.read_excel("Dataset_Consumers.xlsx")
producers_data = pd.read_excel("Dataset_Producers.xlsx")

num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = consumers_data.shape[0]
P_load = consumers_data.sum(axis=1).values  # (T,)
P_production = producers_data.sum(axis=1).values  # (T,)

#%% Variáveis da metaheurística
P_prod_max = np.tile(P_production, (N_batteries, 1))  # (N, T)
P_prod_max_flat = P_prod_max.flatten()
P_disch_max_flat = np.tile(E_max, T)
P_grid_max = max(P_load) * 1.5
P_grid_max_vector = np.full(T, P_grid_max)

#%% Função Objetivo
def denormalize(x, lower, upper):
    return lower + x * (upper - lower)

def objective(x):
    P_c = np.array([
        denormalize(x[i], 0, P_prod_max_flat[i])
        for i in range(N_batteries * T)
    ]).reshape(N_batteries, T)

    P_d = np.array([
        denormalize(x[i + N_batteries * T], 0, P_disch_max_flat[i])
        for i in range(N_batteries * T)
    ]).reshape(N_batteries, T)

    P_grid = np.array([
        denormalize(x[i + 2 * N_batteries * T], 0, P_grid_max)
        for i in range(T)
    ])

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
    strategy='best1bin',
    popsize=5,
    maxiter=50,
    polish=True,
    disp=True,
    updating="deferred"
)

#%% Extrair resultados
x_opt = result.x
P_c_opt = np.array([
    denormalize(x_opt[i], 0, P_prod_max_flat[i])
    for i in range(N_batteries * T)
]).reshape(N_batteries, T)

P_d_opt = np.array([
    denormalize(x_opt[i + N_batteries * T], 0, P_disch_max_flat[i])
    for i in range(N_batteries * T)
]).reshape(N_batteries, T)

P_grid_opt = np.array([
    denormalize(x_opt[i + 2 * N_batteries * T], 0, P_grid_max)
    for i in range(T)
])

#%% Calcular SoC (correto)
SoC = np.zeros((N_batteries, T + 1))
SoC[:, 0] = initial_SoC
P_d_real = np.zeros_like(P_d_opt)

for t in range(T):
    for i in range(N_batteries):
        P_d_real[i, t] = min(P_d_opt[i, t], SoC[i, t])
        SoC[i, t+1] = SoC[i, t] + eta_c[i] * P_c_opt[i, t] - P_d_real[i, t]
        SoC[i, t+1] = np.clip(SoC[i, t+1], 0, E_max[i])

#%% Guardar e visualizar
for i in range(N_batteries):
    df = pd.DataFrame({
        "Time": np.arange(T),
        "P_charge": P_c_opt[i],
        "P_discharge": P_d_real[i],
        "SoC": SoC[i, 1:]
    })
    df.to_csv(f"{output_folder}/battery_{i}_results.csv", index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(df["Time"], df["SoC"], label=f"SoC Bateria {i}", linewidth=2)
    plt.title(f"SoC - Bateria {i}")
    plt.xlabel("Time step")
    plt.ylabel("kWh")
    plt.grid()
    plt.legend()
    plt.savefig(f"{output_folder}/soc_battery_{i}.png")
    plt.close()

#%% Gráfico da rede
plt.figure(figsize=(14, 5))
plt.plot(P_grid_opt, label="Consumo da Rede", linewidth=2)
plt.title("Energia Comprada à Rede")
plt.xlabel("Time step")
plt.ylabel("kWh")
plt.grid()
plt.legend()
plt.savefig(f"{output_folder}/grid_power.png")
plt.close()

#%% Custo final
grid_cost = np.sum(P_grid_opt) * C_grid
degradation_cost = np.sum([battery_degradation_factor[i] * np.sum(P_d_real[i]) for i in range(N_batteries)])
print(f"Custo Total: {grid_cost + degradation_cost:.2f} €")
