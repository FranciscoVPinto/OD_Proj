import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import os

#%% Parameters
E_max = np.array([2500])
eta_c = np.array([0.95])
eta_d = np.array([0.95])
initial_SoC = E_max * 0.5
C_grid = 0.1

# 8-hour blocks
AGGREGATION_INTERVAL = 480  # 8 hours in minutes
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15  # 480/15 = 32

output_folder = "de_plots_8h"
os.makedirs(output_folder, exist_ok=True)

#%% Load Data
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

# Truncate so number of rows is a multiple of TIME_STEP_RATIO
num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

# Aggregate by 8-hour blocks
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

# Use full time series, now in 8-hour blocks
T = consumers_data.shape[0]
P_load_matrix = consumers_data.values.T
P_production_matrix = producers_data.values.T

n_consumers = P_load_matrix.shape[0]
n_producers = P_production_matrix.shape[0]

# Discharge bounds per time step
P_disch_max_flat = np.tile(E_max, T)

# Possible maximum grid import per consumer-block
# This is the maximum consumption each consumer exhibits, repeated for T blocks
P_grid_max = np.tile(P_load_matrix.max(axis=1), T)

#%% Denormalize Function
def denormalize_vector(x, lower, upper):
    """Convert x in [0,1] to [lower,upper]."""
    return lower + x * (upper - lower)

#%% Objective Function
def objective(x):
    """
    Calculates total cost = grid cost + penalties for various violations.
    """
    penalty_weights = {
        "max_power": 200,
        "SoC_violation": 200,
        "over_discharge": 200,
        "fallback_grid": 1e12,
        "grid_priority_violation": 10,
        "unused_production": 100
    }

    # Offsets for slicing x
    offset1 = n_producers * T     # for P_c
    offset2 = offset1 + T         # for P_d
    offset3 = offset2 + n_consumers * T  # for P_grid

    # Denormalize each group of variables
    P_c = denormalize_vector(x[:offset1], 0, P_production_matrix.flatten()).reshape(n_producers, T)
    P_d = denormalize_vector(x[offset1:offset2], 0, P_disch_max_flat).reshape(1, T)
    P_grid = denormalize_vector(x[offset2:offset3], 0, P_grid_max).reshape(n_consumers, T)

    # Initialize SoC
    SoC = np.zeros((1, T + 1))
    SoC[0, 0] = initial_SoC[0]

    penalty = 0.0
    # Example constraints: limit charging/discharging to 25% of E_max per block,
    # enforce a minimum SoC of 10% of capacity
    SoC_min = 0.10 * E_max[0]
    P_c_max = 0.25 * E_max[0]
    P_d_max = 0.25 * E_max[0]

    for t in range(T):
        # --- Limit charging power ---
        P_c_total = np.sum(P_c[:, t])
        if P_c_total > P_c_max:
            penalty += penalty_weights["max_power"] * (P_c_total - P_c_max)
            scale = P_c_max / P_c_total if P_c_total > 0 else 0
            P_c[:, t] *= scale

        # --- Limit discharging power ---
        if P_d[0, t] > P_d_max:
            penalty += penalty_weights["max_power"] * (P_d[0, t] - P_d_max)
            P_d[0, t] = P_d_max

        # --- Over-discharge check ---
        if P_d[0, t] > SoC[0, t]:
            penalty += penalty_weights["over_discharge"] * (P_d[0, t] - SoC[0, t])
            P_d[0, t] = SoC[0, t]

        # --- Update SoC ---
        SoC[0, t+1] = SoC[0, t] + eta_c[0] * P_c_total - P_d[0, t]

        # --- SoC bounds ---
        if SoC[0, t+1] < SoC_min:
            penalty += penalty_weights["SoC_violation"] * (SoC_min - SoC[0, t+1])
            SoC[0, t+1] = SoC_min
        elif SoC[0, t+1] > E_max[0]:
            penalty += penalty_weights["SoC_violation"] * (SoC[0, t+1] - E_max[0])
            SoC[0, t+1] = E_max[0]

        # --- Production bound ---
        for p in range(n_producers):
            if P_c[p, t] > P_production_matrix[p, t]:
                penalty += penalty_weights["max_power"] * (
                    P_c[p, t] - P_production_matrix[p, t]
                )

        # --- Unused production penalty ---
        unused_production = np.sum(P_production_matrix[:, t]) - P_c_total
        if unused_production > 0:
            penalty += penalty_weights["unused_production"] * unused_production

        # --- If battery + grid < consumption => fallback penalty ---
        battery_energy = P_d[0, t] * eta_d[0]
        grid_energy = np.sum(P_grid[:, t])
        total_demand = np.sum(P_load_matrix[:, t])
        supplied_energy = battery_energy + grid_energy
        if supplied_energy < total_demand:
            penalty += penalty_weights["fallback_grid"] * (total_demand - supplied_energy)

        # --- If battery alone could supply total demand, but we used grid => penalty ---
        battery_energy_available = SoC[0, t] * eta_d[0]
        if battery_energy_available >= total_demand and grid_energy > 0:
            penalty += penalty_weights["grid_priority_violation"] * grid_energy

    # --- Objective: sum of grid cost + penalty ---
    return np.sum(P_grid) * C_grid + penalty

#%% Main execution
if __name__ == "__main__":
    # Number of decision variables
    n_vars = n_producers * T + T + n_consumers * T

    # All variables in [0, 1]; real scale done in objective
    bounds = [(0, 1)] * n_vars

    # Solve with Differential Evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy='best1bin',
        mutation=(0.5, 1),
        recombination=0.7,
        popsize=10,
        maxiter=30,
        polish=True,
        disp=True,
        updating="deferred",
        workers=-1
    )

    x_opt = result.x

    # Slice the decision vector
    offset1 = n_producers * T
    offset2 = offset1 + T
    offset3 = offset2 + n_consumers * T

    # Reconstruct the optimized arrays
    P_c_opt = denormalize_vector(x_opt[:offset1], 0,
                                 P_production_matrix.flatten()).reshape(n_producers, T)
    P_d_opt = denormalize_vector(x_opt[offset1:offset2], 0,
                                 P_disch_max_flat).reshape(1, T)
    P_grid_opt = denormalize_vector(x_opt[offset2:offset3], 0,
                                    P_grid_max).reshape(n_consumers, T)

    # Compute SoC with feasible discharge
    SoC = np.zeros((1, T + 1))
    SoC[0, 0] = initial_SoC[0]
    P_d_real = np.zeros_like(P_d_opt)

    for t in range(T):
        # Respect actual SoC
        P_d_real[0, t] = min(P_d_opt[0, t], SoC[0, t])
        SoC[0, t+1] = SoC[0, t] + eta_c[0] * np.sum(P_c_opt[:, t]) - P_d_real[0, t]
        SoC[0, t+1] = np.clip(SoC[0, t+1], 0, E_max[0])

    # Summaries
    P_charge_total = np.sum(P_c_opt, axis=0)
    P_discharge_total = P_d_real[0]
    Battery_SoC_total = SoC[0, 1:]
    P_grid_total = np.sum(P_grid_opt, axis=0)

    # Build results DataFrame
    results_df = pd.DataFrame({
        "Time": np.arange(T),
        "P_grid": P_grid_total,
        "P_charge": P_charge_total,
        "P_discharge": P_discharge_total,
        "Battery_SoC": Battery_SoC_total
    })
    results_df["Cost"] = results_df["P_grid"] * C_grid
    results_df.to_csv(os.path.join(output_folder, "de_8h_results.csv"), index=False)

    #%% Plot Results

    # Battery SoC
    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Time"], results_df["Battery_SoC"],
             label="Battery SoC (kWh)", linestyle='dashed', linewidth=2)
    plt.xlabel("Time Step (8H intervals)")
    plt.ylabel("Battery Energy (kWh)")
    plt.title("Battery SoC Over Time (8-hour Blocks)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time_8h.png"))
    plt.show()

    # Charging vs. Discharging
    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Time"], results_df["P_charge"], label="Battery Charging (kW)",
             linestyle='dashed', linewidth=2)
    plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharging (kW)",
             linestyle='solid', linewidth=2)
    plt.xlabel("Time Step (8H intervals)")
    plt.ylabel("Power (kW)")
    plt.title("Charging and Discharging Patterns (8-hour Blocks)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "Charging_Discharging_8h.png"))
    plt.show()

    print(f"\n✅ Final cost with penalties: {result.fun:.2f} €")
    print(f"💡 Grid cost only (no penalties): {results_df['Cost'].sum():.2f} €")

    #%% Plot: Consumption, Battery Discharge, Grid, and Production
    total_consumption = np.sum(P_load_matrix, axis=0)
    total_production = np.sum(P_production_matrix, axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Time"], total_consumption, label="Total Consumption (kW)",
             linewidth=2)
    plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharge (kW)",
             linestyle='dashed', linewidth=2)
    plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Import (kW)",
             linestyle='dashdot', linewidth=2)
    plt.plot(results_df["Time"], total_production, label="Total Production (kW)",
             linestyle='dotted', linewidth=2)

    plt.xlabel("Time Step (8H intervals)")
    plt.ylabel("Power (kW)")
    plt.title("Energy Flows Over Time (8-hour Blocks)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "Energy_Flows_Over_Time_8h.png"))
    plt.show()
