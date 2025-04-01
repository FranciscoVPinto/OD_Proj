import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import differential_evolution

#%% Parâmetros fixos
E_max = 2500
eta_c = 0.95
eta_d = 0.95
initial_SoC = E_max * 0.5
C_grid = 0.1
lambda_waste = 0.2

AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15

output_folder = "de_1p1c_plots"
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
P_load = consumers_data.sum(axis=1).values
P_prod = producers_data.sum(axis=1).values

#%% Bounds
bounds = [(0.0, prod) for prod in P_prod] + [(0.0, E_max) for _ in range(T)]

# %% Função Objetivo
def objective(x):
    P_charge = x[:T]
    P_discharge = x[T:]
    SoC = np.zeros(T + 1)
    SoC[0] = initial_SoC
    grid_cost, waste_cost, penalty = 0.0, 0.0, 0.0

    for t in range(T):
        SoC[t+1] = SoC[t] + eta_c * P_charge[t] - P_discharge[t]
        SoC[t+1] = np.clip(SoC[t+1], 0, E_max)

        grid = max(0.0, P_load[t] - eta_d * P_discharge[t])
        waste = max(0.0, P_prod[t] - P_charge[t])

        grid_cost += grid * C_grid
        waste_cost += waste * lambda_waste

        if P_charge[t] > 0 and P_discharge[t] > 0:
            penalty += 1e4 * min(P_charge[t], P_discharge[t])

    penalty += 1e4 * abs(SoC[-1] - initial_SoC)
    penalty += 1e5 * np.sum(np.maximum(0.0, P_charge - P_prod))

    energia_carregada_util = np.sum(P_charge) * eta_c
    energia_total_descarga = np.sum(P_discharge)
    delta_soc = SoC[-1] - SoC[0]
    energia_total_disponivel = energia_carregada_util + delta_soc

    if energia_total_descarga - energia_total_disponivel > 1e-3:
        return 1e12

    return grid_cost + waste_cost + penalty

# %% Gerar população inicial válida
popsize = 15
init_population = []
while len(init_population) < popsize:
    P_c = np.minimum(P_prod, np.random.uniform(0.2, 0.6, size=T) * P_prod)
    P_d = np.minimum(P_c * eta_c / eta_d, np.random.uniform(0.2, 0.6, size=T) * E_max)

    SoC = np.zeros(T + 1)
    SoC[0] = initial_SoC
    for t in range(T):
        SoC[t+1] = np.clip(SoC[t] + eta_c * P_c[t] - P_d[t], 0, E_max)

    energia_carregada_util = np.sum(P_c) * eta_c
    energia_total_descarga = np.sum(P_d)
    delta_soc = SoC[-1] - SoC[0]
    energia_total_disponivel = energia_carregada_util + delta_soc

    if energia_total_descarga <= energia_total_disponivel + 1e-3:
        init_population.append(np.concatenate([P_c, P_d]))
init_population = np.array(init_population)

# %% Executar o algoritmo DE
result = differential_evolution(
    objective,
    bounds=bounds,
    strategy='best1bin',
    mutation=(0.5, 1.0),
    recombination=0.7,
    popsize=popsize,
    maxiter=1000,
    polish=False,
    init=init_population,
    disp=True
)

# %% Extrair resultados
x_opt = result.x
P_charge_opt = x_opt[:T]
P_discharge_opt = x_opt[T:]

SoC = np.zeros(T + 1)
SoC[0] = initial_SoC
P_grid = np.zeros(T)
P_waste = np.zeros(T)

for t in range(T):
    SoC[t+1] = np.clip(SoC[t] + eta_c * P_charge_opt[t] - P_discharge_opt[t], 0, E_max)
    P_grid[t] = max(0.0, P_load[t] - eta_d * P_discharge_opt[t])
    P_waste[t] = max(0.0, P_prod[t] - P_charge_opt[t])

# %% Guardar resultados
results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_charge": P_charge_opt,
    "P_discharge": P_discharge_opt,
    "Battery_SoC": SoC[1:],
    "P_grid": P_grid,
    "P_waste": P_waste,
    "Cost_grid": P_grid * C_grid,
    "Energy_waste": P_waste,
})
results_df.to_csv(os.path.join(output_folder, "Simplified_DE_results.csv"), index=False)

# %% Gerar gráficos
plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="SoC (kWh)")
plt.title("Battery State of Charge Over Time")
plt.xlabel("8h Time Blocks")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "SoC_Simplified.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], results_df["P_charge"], label="Charge (kW)")
plt.plot(results_df["Time"], results_df["P_discharge"], label="Discharge (kW)")
plt.title("Charging / Discharging")
plt.xlabel("8h Time Blocks")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Charge_Discharge_Simplified.png"))
plt.show()

plt.figure(figsize=(14,6))
plt.plot(results_df["Time"], P_load, label="Load", linewidth=2)
plt.plot(results_df["Time"], P_discharge_opt, label="Battery Discharge", linestyle='--')
plt.plot(results_df["Time"], P_grid, label="Grid Import", linestyle='-.')
plt.plot(results_df["Time"], P_prod, label="Production", linestyle=':')
plt.title("Energy Flow Over Time")
plt.xlabel("8h Time Blocks")
plt.grid()
plt.legend()
plt.savefig(os.path.join(output_folder, "Flows_Simplified.png"))
plt.show()

print(f"\n Total grid cost: {results_df['Cost_grid'].sum():.2f} €")
