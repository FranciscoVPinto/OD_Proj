import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso
import os

#%% Parâmetros fixos
E_max = 2500
eta_c = 0.95
eta_d = 0.95
initial_SoC = E_max * 0.5
C_grid = 0.1
PENALIZE_GRID = 2.0
PENALIZE_WASTE = 1.5

AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15
output_folder = "pso_14p51c_plots"
os.makedirs(output_folder, exist_ok=True)

#%% Carregar dados
consumers = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

nrows = (len(consumers) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers = consumers.iloc[:nrows]
producers = producers.iloc[:nrows]

consumers = consumers.groupby(consumers.index // TIME_STEP_RATIO).sum()
producers = producers.groupby(producers.index // TIME_STEP_RATIO).sum()

P_load = consumers.sum(axis=1).values
P_production = producers.sum(axis=1).values
T = len(P_load)

#%% Variáveis de decisão: P_charge, P_discharge
n_vars = 2 * T
lb = np.zeros(n_vars)
ub = np.ones(n_vars)

def denormalize_vector(x, lower, upper):
    return lower + x * (upper - lower)

#%% Função Objetivo

def objective(x):
    x_c = x[:T]
    x_d = x[T:]

    P_c = np.array([denormalize_vector(x_c[t], 0, P_production[t]) for t in range(T)])
    P_d = denormalize_vector(x_d, 0, E_max)

    SoC = np.zeros(T + 1)
    SoC[0] = initial_SoC
    total_cost = 0.0

    for t in range(T):
        real_P_d = min(P_d[t], SoC[t])
        battery_supply = eta_d * real_P_d

        P_c_max = min(P_production[t], (E_max - SoC[t]) / eta_c)
        P_c[t] = min(P_c[t], P_c_max)

        SoC[t+1] = SoC[t] + eta_c * P_c[t] - real_P_d / eta_d
        SoC[t+1] = np.clip(SoC[t+1], 0, E_max)

        prod_used = P_c[t]
        P_waste = max(0, P_production[t] - prod_used)

        total_available = battery_supply + prod_used
        P_grid = max(0, P_load[t] - total_available)
        total_supply = battery_supply + prod_used + P_grid

        # ⚠️ Penalização forte se não for fisicamente válido
        if abs(total_supply - P_load[t]) > 1e-3:
            return 1e10

        total_cost += P_grid * (C_grid + PENALIZE_GRID)
        total_cost += P_waste * PENALIZE_WASTE

    return total_cost

#%% Executar PSO
x_opt, fopt = pso(
    objective,
    lb,
    ub,
    swarmsize=300,
    maxiter=600,
    minstep=1e-6,
    debug=True
)

#%% Extrair resultados
x_c = x_opt[:T]
x_d = x_opt[T:]
P_c = np.array([denormalize_vector(x_c[t], 0, P_production[t]) for t in range(T)])
P_d = denormalize_vector(x_d, 0, E_max)

SoC = np.zeros(T + 1)
SoC[0] = initial_SoC
P_d_real = np.zeros(T)
P_grid = np.zeros(T)
P_waste = np.zeros(T)
total_supply = np.zeros(T)

for t in range(T):
    P_d_real[t] = min(P_d[t], SoC[t])
    battery_supply = eta_d * P_d_real[t]

    P_c_max = min(P_production[t], (E_max - SoC[t]) / eta_c)
    P_c[t] = min(P_c[t], P_c_max)

    SoC[t+1] = SoC[t] + eta_c * P_c[t] - P_d_real[t] / eta_d
    SoC[t+1] = np.clip(SoC[t+1], 0, E_max)

    prod_used = P_c[t]
    total_available = battery_supply + prod_used
    P_grid[t] = max(0, P_load[t] - total_available)
    total_supply[t] = prod_used + battery_supply + P_grid[t]
    P_waste[t] = P_production[t] - prod_used

#%% Guardar resultados
df = pd.DataFrame({
    "Time": np.arange(T),
    "P_load": P_load,
    "P_production": P_production,
    "P_charge": P_c,
    "P_discharge": P_d_real,
    "P_grid": P_grid,
    "P_waste": P_waste,
    "Battery_SoC": SoC[1:],
    "Total_Supply": total_supply
})
df["Cost"] = df["P_grid"] * C_grid
df.to_csv(os.path.join(output_folder, "pso_results_final_sem_violacoes.csv"), index=False)

#%% Gráficos
plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["Battery_SoC"], label="Battery SoC (kWh)", linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (4H intervals)")
plt.ylabel("Energy (kWh)")
plt.title("Battery State of Charge Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["P_charge"], label="Battery Charging (kW)", linestyle='dashed')
plt.plot(df["Time"], df["P_discharge"], label="Battery Discharging (kW)", linestyle='solid')
plt.xlabel("Time Step")
plt.ylabel("Power (kW)")
plt.title("Charging and Discharging")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Charge_Discharge.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df["Time"], df["P_load"], label="Load", linewidth=2)
plt.plot(df["Time"], df["P_discharge"], label="Battery Discharge", linestyle='dashed', linewidth=2)
plt.plot(df["Time"], df["P_grid"], label="Grid", linestyle='dashdot', linewidth=2)
plt.plot(df["Time"], df["P_production"], label="Production", linestyle='dotted', linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Power (kW)")
plt.title("Energy Flows Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Energy_Flows.png"))
plt.show()

print(f"Total Grid Cost (PSO): {df['Cost'].sum():.2f} €")
