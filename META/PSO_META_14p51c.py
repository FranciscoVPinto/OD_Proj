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


lb = np.zeros(T * (N_producers + N_consumers))
ub = [max(1e-3, P_prod_matrix[t, j]) for t in range(T) for j in range(N_producers)] + [E_max / N_consumers] * T * N_consumers
ub = np.array(ub)

#%% Função Objetivo

def objective(x):
    idx1 = T * N_producers
    P_charge_matrix = x[:idx1].reshape((T, N_producers))
    P_discharge_matrix = x[idx1:].reshape((T, N_consumers))
    P_charge_total = P_charge_matrix.sum(axis=1)

    SoC = np.zeros((T + 1, N_consumers))
    SoC[0, :] = initial_SoC / N_consumers

    grid_cost, waste_cost, penalty = 0.0, 0.0, 0.0

    for t in range(T):
        
        excess_charge = np.sum(np.maximum(0.0, P_charge_matrix[t] - P_prod_matrix[t]))
        penalty += 1e6 * excess_charge

        real_discharge = np.minimum(P_discharge_matrix[t], SoC[t])
        SoC[t+1] = np.clip(SoC[t] + (eta_c * P_charge_total[t] / N_consumers) - real_discharge, 0, E_max / N_consumers)

        available_energy = eta_d * np.sum(real_discharge)
        total_load = P_load_matrix[t].sum()
        P_grid = max(0.0, total_load - available_energy)
        grid_cost += P_grid * C_grid

        penalty += 1e6 * abs(P_grid + available_energy - total_load)

        wasted_prod = np.maximum(0.0, P_prod_matrix[t] - P_charge_matrix[t])
        waste_cost += np.sum(wasted_prod) * lambda_waste

        if np.sum(P_discharge_matrix[t]) > 0 and P_charge_total[t] > 0:
            penalty += 1e5 * min(np.sum(P_discharge_matrix[t]), P_charge_total[t])

    final_soc_total = np.sum(SoC[-1])
    initial_soc_total = np.sum(SoC[0])
    penalty += 1e4 * abs(final_soc_total - initial_soc_total)

    energia_carregada_util = np.sum(P_charge_total) * eta_c
    energia_total_descarga = np.sum(real_discharge)
    delta_soc = final_soc_total - initial_soc_total
    energia_total_disponivel = energia_carregada_util + delta_soc

    if energia_total_descarga - energia_total_disponivel > 1e-3:
        return 1e12

    return grid_cost + waste_cost + penalty

#%% PSO
x_opt, fopt = pso(objective, lb, ub, swarmsize=200, maxiter=150, minstep=1e-6, debug=True)

#%% Resultados
idx1 = T * N_producers
P_charge_matrix_opt = x_opt[:idx1].reshape((T, N_producers))
P_discharge_matrix_opt = x_opt[idx1:].reshape((T, N_consumers))
P_charge_total = P_charge_matrix_opt.sum(axis=1)
P_discharge_total = P_discharge_matrix_opt.sum(axis=1)

SoC = np.zeros((T + 1, N_consumers))
SoC[0, :] = initial_SoC / N_consumers
P_grid = np.zeros(T)
P_waste = np.zeros(T)
P_discharge_real = np.zeros(T)

for t in range(T):
    real_discharge = np.minimum(P_discharge_matrix_opt[t], SoC[t])
    SoC[t+1] = np.clip(SoC[t] + (eta_c * P_charge_total[t] / N_consumers) - real_discharge, 0, E_max / N_consumers)
    P_discharge_real[t] = np.sum(real_discharge)
    available_energy = eta_d * P_discharge_real[t]
    total_load = P_load_matrix[t].sum()
    P_grid[t] = max(0.0, total_load - available_energy)
    P_waste[t] = np.sum(np.maximum(0.0, P_prod_matrix[t] - P_charge_matrix_opt[t]))

#%% Exportar resultados
results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_charge_total": P_charge_total,
    "P_discharge_total": P_discharge_real,
    "Battery_SoC_avg": SoC[1:].mean(axis=1),
    "P_grid": P_grid,
    "P_waste": P_waste,
    "Cost_grid": P_grid * C_grid
})

results_df.to_csv(os.path.join(output_folder, "PSO_MultiAgent_PerConsumer_Discharge.csv"), index=False)

#%% Gráficos
plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["Battery_SoC_avg"], label="SoC Média (kWh)")
plt.title("Estado de Carga da Bateria - Média por Consumidor")
plt.xlabel("Blocos de tempo de 8h")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Battery_SoC.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["P_charge_total"], label="Carga (kW)")
plt.plot(results_df["Time"], results_df["P_discharge_total"], label="Descarga Total (kW)")
plt.title("Carga / Descarga por Consumidor")
plt.xlabel("Blocos de tempo de 8h")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Charge_Discharge.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], P_load_matrix.sum(axis=1), label="Carga Total", linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge_total"], label="Descarga da Bateria", linestyle='--')
plt.plot(results_df["Time"], P_grid, label="Importação da Rede", linestyle='-.')
plt.plot(results_df["Time"], P_prod_matrix.sum(axis=1), label="Produção", linestyle=':' )
plt.title("Fluxo de Energia ao Longo do Tempo")
plt.xlabel("Blocos de tempo de 8h")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Energy_Flow.png"))
plt.show()

print(f"\nCusto total com a rede: {results_df['Cost_grid'].sum():.2f} €")
