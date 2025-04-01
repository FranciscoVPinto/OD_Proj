import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pyswarm import pso

#%% Parâmetros
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

#%% Carregar dados
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

# Variáveis: carga por produtor, descarga por consumidor
lb = np.zeros(T * (N_producers + N_consumers))
ub = [max(1e-3, P_prod_matrix[t, j]) for t in range(T) for j in range(N_producers)] + [E_max] * T * N_consumers
ub = np.array(ub)

#%% Função Objetivo

def objective(x):
    idx1 = T * N_producers
    P_charge_matrix = x[:idx1].reshape((T, N_producers))
    P_discharge_matrix = x[idx1:].reshape((T, N_consumers))
    P_charge_total = P_charge_matrix.sum(axis=1)
    P_discharge_total = P_discharge_matrix.sum(axis=1)

    SoC = np.zeros(T + 1)
    SoC[0] = initial_SoC
    grid_cost, waste_cost, penalty = 0.0, 0.0, 0.0

    for t in range(T):
        excess_charge = np.sum(np.maximum(0.0, P_charge_matrix[t] - P_prod_matrix[t]))
        penalty += 1e6 * excess_charge

        real_discharge = min(P_discharge_total[t], SoC[t])
        SoC[t+1] = np.clip(SoC[t] + eta_c * P_charge_total[t] - real_discharge, 0, E_max)

        available_energy = eta_d * real_discharge
        total_load = P_load_matrix[t].sum()
        P_grid = max(0.0, total_load - available_energy)
        grid_cost += P_grid * C_grid

        penalty += 1e6 * abs(P_grid + available_energy - total_load)

        wasted_prod = np.maximum(0.0, P_prod_matrix[t] - P_charge_matrix[t])
        waste_cost += np.sum(wasted_prod) * lambda_waste

        if P_discharge_total[t] > 0 and P_charge_total[t] > 0:
            penalty += 1e5 * min(P_discharge_total[t], P_charge_total[t])

    penalty += 1e4 * abs(SoC[-1] - initial_SoC)

    energia_carregada_util = np.sum(P_charge_total) * eta_c
    energia_total_descarga = np.sum([min(P_discharge_total[t], SoC[t]) for t in range(T)])
    delta_soc = SoC[-1] - SoC[0]
    energia_total_disponivel = energia_carregada_util + delta_soc

    if energia_total_descarga - energia_total_disponivel > 1e-3:
        return 1e12

    return grid_cost + waste_cost + penalty

#%% PSO
x_opt, fopt = pso(objective, lb, ub, swarmsize=250, maxiter=150, minstep=1e-6, debug=True)

#%% Pós-processamento dos resultados
idx1 = T * N_producers
P_charge_matrix_opt = x_opt[:idx1].reshape((T, N_producers))
P_discharge_matrix_opt = x_opt[idx1:].reshape((T, N_consumers))
P_charge_total = P_charge_matrix_opt.sum(axis=1)
P_discharge_total = P_discharge_matrix_opt.sum(axis=1)

SoC = np.zeros(T + 1)
SoC[0] = initial_SoC
P_grid = np.zeros(T)
P_waste = np.zeros(T)
P_discharge_real = np.zeros(T)

for t in range(T):
    P_discharge_real[t] = min(P_discharge_total[t], SoC[t])
    SoC[t+1] = np.clip(SoC[t] + eta_c * P_charge_total[t] - P_discharge_real[t], 0, E_max)
    available_energy = eta_d * P_discharge_real[t]
    total_load = P_load_matrix[t].sum()
    P_grid[t] = max(0.0, total_load - available_energy)
    P_waste[t] = np.sum(np.maximum(0.0, P_prod_matrix[t] - P_charge_matrix_opt[t]))

#%% Salvar resultados
results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_charge_total": P_charge_total,
    "P_discharge": P_discharge_real,
    "Battery_SoC": SoC[1:],
    "P_waste": P_waste,
    "Cost_grid": P_grid,
    "Energy_waste": P_waste
})

results_df.to_csv(os.path.join(output_folder, "PSO_MultiAgent_Individual_Results.csv"), index=False)

#%% Gráficos
plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="SoC (kWh)")
plt.title("Estado de carga da bateria ao longo do tempo")
plt.xlabel("Blocos de tempo de 8h")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Battery_SoC.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["P_charge_total"], label="Carga (kW)")
plt.plot(results_df["Time"], results_df["P_discharge"], label="Descarga (kW)")
plt.title("Carga / Descarga")
plt.xlabel("Blocos de tempo de 8h")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Charge_Discharge.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], P_load_matrix.sum(axis=1), label="Carga Total", linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Descarga Bateria", linestyle='--')
plt.plot(results_df["Time"], P_grid, label="Importação da Rede", linestyle='-.')
plt.plot(results_df["Time"], P_prod_matrix.sum(axis=1), label="Produção", linestyle=':' )
plt.title("Fluxo de Energia ao longo do tempo")
plt.xlabel("Blocos de tempo de 8h")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Energy_Flow.png"))
plt.show()

print(f"\n Custo total com a rede: {results_df['Cost_grid'].sum():.2f} €")