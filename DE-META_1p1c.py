# %% Simplified DE Optimization with Aggregated Producer & Consumer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import differential_evolution

# Parameters
E_max = 2500
eta_c = 0.95
eta_d = 0.95
initial_SoC = E_max * 0.5
C_grid = 0.1           # €/kWh
lambda_waste = 0.2     # €/kWh

AGGREGATION_INTERVAL = 480  # 8h
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15

output_folder = "de_1p1c_plots"
os.makedirs(output_folder, exist_ok=True)

# Load data
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

# Aggregate to 8h blocks
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = len(consumers_data)

P_load = consumers_data.sum(axis=1).values     # shape: (T,)
P_prod = producers_data.sum(axis=1).values     # shape: (T,)

# Decision vars: [P_charge_0 ... P_charge_T-1, P_discharge_0 ... P_discharge_T-1]
bounds = [(0.0, prod) for prod in P_prod] + [(0.0, E_max) for _ in range(T)]

def objective(x):
    P_charge = x[:T]
    P_discharge = x[T:]

    SoC = np.zeros(T + 1)
    SoC[0] = initial_SoC

    grid_cost = 0.0
    waste_cost = 0.0
    penalty = 0.0

    for t in range(T):
        # Update SoC
        SoC[t+1] = SoC[t] + eta_c * P_charge[t] - P_discharge[t]

        # Clamp SoC within bounds
        if SoC[t+1] < 0:
            penalty += 1e4 * abs(SoC[t+1])
            SoC[t+1] = 0
        elif SoC[t+1] > E_max:
            penalty += 1e4 * (SoC[t+1] - E_max)
            SoC[t+1] = E_max

        # Grid import needed if battery not enough
        P_available = eta_d * P_discharge[t]
        grid = max(0.0, P_load[t] - P_available)

        # Wasted production if not stored
        waste = max(0.0, P_prod[t] - P_charge[t])

        grid_cost += grid * C_grid
        waste_cost += waste * lambda_waste

    total_cost = grid_cost + waste_cost + penalty
    return total_cost

# Run DE
result = differential_evolution(
    objective,
    bounds=bounds,
    strategy='best1bin',
    mutation=(0.5, 1.0),
    recombination=0.7,
    popsize=15,
    maxiter=5,
    polish=True,
    disp=True,
    updating="deferred"
)

x_opt = result.x
P_charge_opt = x_opt[:T]
P_discharge_opt = x_opt[T:]

# Compute SoC, Grid, Waste for optimal
SoC = np.zeros(T + 1)
SoC[0] = initial_SoC
P_grid = np.zeros(T)
P_waste = np.zeros(T)

for t in range(T):
    SoC[t+1] = SoC[t] + eta_c * P_charge_opt[t] - P_discharge_opt[t]
    SoC[t+1] = np.clip(SoC[t+1], 0, E_max)

    P_grid[t] = max(0.0, P_load[t] - eta_d * P_discharge_opt[t])
    P_waste[t] = max(0.0, P_prod[t] - P_charge_opt[t])

# Save results
results_df = pd.DataFrame({
    "Time": np.arange(T),
    "P_charge": P_charge_opt,
    "P_discharge": P_discharge_opt,
    "Battery_SoC": SoC[1:],
    "P_grid": P_grid,
    "P_waste": P_waste,
    "Cost_grid": P_grid * C_grid,
    "Cost_waste": P_waste * lambda_waste,
})

results_df.to_csv(os.path.join(output_folder, "Simplified_DE_results.csv"), index=False)

# Plot
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

print(f"💡 Grid cost: {results_df['Cost_grid'].sum():.2f} €")
