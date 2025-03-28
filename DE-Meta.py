import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import differential_evolution, NonlinearConstraint

#%% Parameters
N_batteries = 1
E_max = np.array([2500])  # Battery capacity
eta_c = np.array([0.95])  # Charging efficiency
eta_d = np.array([0.95])  # Discharging efficiency
initial_SoC = E_max * 0.5 # 50% SoC initially
C_grid = 0.1              # Grid cost (€/kWh)


AGGREGATION_INTERVAL = 480  # 8 hours in minutes
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15  # 480/15 = 32

output_folder = "de_plots"
os.makedirs(output_folder, exist_ok=True)

#%% Load Data
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

# Truncate to make the length a multiple of TIME_STEP_RATIO
num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

# Aggregate by 8-hour blocks
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = consumers_data.shape[0]
P_load_matrix = consumers_data.values.T       # shape: (n_consumers, T)
P_production_matrix = producers_data.values.T # shape: (n_producers, T)

n_consumers = P_load_matrix.shape[0]
n_producers = P_production_matrix.shape[0]

# Large upper bound on grid import (like the PSO code). E.g., 1e6:
P_grid_max = np.full(n_consumers * T, 1e6)

# Discharge bounds: let the battery discharge up to its entire capacity each block
P_disch_max_flat = np.tile(E_max, T)

#------------------------------------------------------------------------------
#  Utility: Denormalize variables x from [0, 1] to [lower, upper].
#------------------------------------------------------------------------------
def denormalize_vector(x, lower, upper):
    """Convert x in [0,1] to [lower,upper]."""
    return lower + x * (upper - lower)

#------------------------------------------------------------------------------
#  Objective Function
#------------------------------------------------------------------------------
def objective(x):
    """
    Minimize total grid cost plus penalties.
    Same penalty approach as the PSO script.
    """
    penalty_weights = {
        "max_power":     200,
        "SoC_violation": 200,
        "over_discharge": 200,
    }

    # Offsets for slicing x:
    # 1) P_c (charge from each producer)
    offset1 = n_producers * T
    # 2) P_d (battery discharge)
    offset2 = offset1 + T
    # 3) P_grid
    offset3 = offset2 + n_consumers * T

    # Denormalize each group of variables
    P_c = denormalize_vector(
        x[:offset1],
        0,
        P_production_matrix.flatten()
    ).reshape(n_producers, T)

    P_d = denormalize_vector(
        x[offset1:offset2],
        0,
        P_disch_max_flat
    ).reshape(1, T)

    P_grid = denormalize_vector(
        x[offset2:offset3],
        0,
        P_grid_max
    ).reshape(n_consumers, T)

    # Track battery SoC
    SoC = np.zeros((N_batteries, T + 1))
    SoC[0, 0] = initial_SoC[0]

    # We'll allow SoC_min = 0, SoC_max = E_max
    SoC_min = 0.0
    SoC_max = E_max[0]

    penalty = 0.0

    for t in range(T):
        # 1) Sum of all producer charge
        P_c_total = np.sum(P_c[:, t])

        # 2) Check if we exceed production available
        if P_c_total > np.sum(P_production_matrix[:, t]):
            # Penalty for going above actual production
            penalty += penalty_weights["max_power"] * (
                P_c_total - np.sum(P_production_matrix[:, t])
            )

        # 3) Over-discharge check
        if P_d[0, t] > SoC[0, t]:
            # Battery discharge can’t exceed current SoC
            excess = P_d[0, t] - SoC[0, t]
            penalty += penalty_weights["over_discharge"] * excess
            # clip to SoC
            P_d[0, t] = SoC[0, t]

        # 4) Update SoC
        SoC[0, t+1] = SoC[0, t] + eta_c[0] * P_c_total - P_d[0, t]

        # 5) Check SoC bounds
        if SoC[0, t+1] < SoC_min:
            diff = SoC_min - SoC[0, t+1]
            penalty += penalty_weights["SoC_violation"] * diff
            SoC[0, t+1] = SoC_min

        if SoC[0, t+1] > SoC_max:
            diff = SoC[0, t+1] - SoC_max
            penalty += penalty_weights["SoC_violation"] * diff
            SoC[0, t+1] = SoC_max

    # Grid cost
    grid_cost = np.sum(P_grid) * C_grid

    return grid_cost + penalty

#------------------------------------------------------------------------------
#  Hard Constraint: battery discharge + grid = consumption (per time step)
#------------------------------------------------------------------------------
def constraint_residual(x):
    """
    Return array of length T with supply-demand mismatch at each time step.
    We want: battery_discharge + grid_energy - load = 0.
    We'll enforce this in [-epsilon, +epsilon].
    """
    epsilon = 1e-4

    offset1 = n_producers * T
    offset2 = offset1 + T
    offset3 = offset2 + n_consumers * T

    # Denormalize
    P_c = denormalize_vector(
        x[:offset1],
        0,
        P_production_matrix.flatten()
    ).reshape(n_producers, T)

    P_d = denormalize_vector(
        x[offset1:offset2],
        0,
        P_disch_max_flat
    ).reshape(1, T)

    P_grid = denormalize_vector(
        x[offset2:offset3],
        0,
        P_grid_max
    ).reshape(n_consumers, T)

    # The mismatch = (discharge * eta_d) + grid - load
    mismatch = np.zeros(T)
    for t in range(T):
        battery_energy_t = P_d[0, t] * eta_d[0]
        grid_energy_t    = np.sum(P_grid[:, t])
        total_demand_t   = np.sum(P_load_matrix[:, t])

        mismatch[t] = battery_energy_t + grid_energy_t - total_demand_t

    # Return the mismatch. We'll constrain it in [-epsilon, +epsilon].
    return mismatch

# Prepare the constraint for SciPy's NonlinearConstraint
eps = 1e-4
nonlin_con = NonlinearConstraint(
    fun=constraint_residual,
    lb=-eps,
    ub=+eps
)

#------------------------------------------------------------------------------
#  Solve with Differential Evolution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Number of decision variables
    n_vars = (n_producers * T) + T + (n_consumers * T)

    # Each variable in [0, 1]; real scaling is handled by denormalize_vector
    bounds = [(0.0, 1.0)] * n_vars

    # Run Differential Evolution
    result = differential_evolution(
        objective,
        bounds=bounds,
        constraints=(nonlin_con,),
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        popsize=15,
        maxiter=300,
        polish=True,
        disp=True,
        updating="deferred",
        workers=-1
    )

    x_opt = result.x
    f_opt = result.fun

    #------------------------------------------------------------------------------
    #  Reconstruct the optimized arrays
    #------------------------------------------------------------------------------
    offset1 = n_producers * T
    offset2 = offset1 + T
    offset3 = offset2 + n_consumers * T

    P_c_opt = denormalize_vector(
        x_opt[:offset1],
        0,
        P_production_matrix.flatten()
    ).reshape(n_producers, T)

    P_d_opt = denormalize_vector(
        x_opt[offset1:offset2],
        0,
        P_disch_max_flat
    ).reshape(1, T)

    P_grid_opt = denormalize_vector(
        x_opt[offset2:offset3],
        0,
        P_grid_max
    ).reshape(n_consumers, T)

    #------------------------------------------------------------------------------
    #  Compute the feasible SoC and actual discharge (post-solution check)
    #------------------------------------------------------------------------------
    SoC = np.zeros((1, T + 1))
    SoC[0, 0] = initial_SoC[0]
    P_d_real = np.zeros_like(P_d_opt)

    for t in range(T):
        # Actual feasible discharge can't exceed SoC
        P_d_real[0, t] = min(P_d_opt[0, t], SoC[0, t])
        SoC[0, t+1] = SoC[0, t] + eta_c[0]*np.sum(P_c_opt[:, t]) - P_d_real[0, t]
        SoC[0, t+1] = np.clip(SoC[0, t+1], 0, E_max[0])

    # Summaries
    P_charge_total     = np.sum(P_c_opt, axis=0)
    P_discharge_total  = P_d_real[0]
    Battery_SoC_total  = SoC[0, 1:]
    P_grid_total       = np.sum(P_grid_opt, axis=0)

    results_df = pd.DataFrame({
        "Time":         np.arange(T),
        "P_grid":       P_grid_total,
        "P_charge":     P_charge_total,
        "P_discharge":  P_discharge_total,
        "Battery_SoC":  Battery_SoC_total
    })
    results_df["Cost"] = results_df["P_grid"] * C_grid
    csv_path = os.path.join(output_folder, "DE_results.csv")
    results_df.to_csv(csv_path, index=False)

    #------------------------------------------------------------------------------
    #  Quick Feasibility Check
    #------------------------------------------------------------------------------
    violations = []
    for t in range(T):
        # supply-demand check
        supply = (P_discharge_total[t] * eta_d[0]) + P_grid_total[t]
        demand = np.sum(P_load_matrix[:, t])
        diff = supply - demand
        if abs(diff) > 1e-3:
            violations.append(f"Time {t}: mismatch = {diff:.4f}")

        # SoC check
        if SoC[0, t] < -1e-6:
            violations.append(f"Time {t}: negative SoC = {SoC[0, t]:.4f}")
        if SoC[0, t] > E_max[0] + 1e-6:
            violations.append(f"Time {t}: SoC above capacity = {SoC[0, t]:.4f}")

    if violations:
        print("\n⚠️ Violations in final solution:")
        for v in violations:
            print(" ", v)
    else:
        print("\n✅ All constraints satisfied (battery + grid == load, SoC in [0, E_max]).")

    #------------------------------------------------------------------------------
    #  Plotting
    #------------------------------------------------------------------------------
    # Battery SoC
    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Time"], results_df["Battery_SoC"], 
             label="Battery SoC (kWh)", linestyle='dashed', linewidth=2)
    plt.xlabel("Time Step (8H intervals)")
    plt.ylabel("Battery Energy (kWh)")
    plt.title("Battery SoC Over Time (8-hour Blocks)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time.png"))
    plt.show()

    # Charging vs. Discharging
    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Time"], results_df["P_charge"], 
             label="Battery Charging (kW)", linestyle='dashed', linewidth=2)
    plt.plot(results_df["Time"], results_df["P_discharge"],
             label="Battery Discharging (kW)", linestyle='solid', linewidth=2)
    plt.xlabel("Time Step (8H intervals)")
    plt.ylabel("Power (kW)")
    plt.title("Charging and Discharging Patterns (8-hour Blocks)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "Charging_Discharging.png"))
    plt.show()

    # Print final results
    print(f"\n✅ DE final objective (cost + penalties): {f_opt:.2f} €")
    print(f"💡 Grid cost only: {results_df['Cost'].sum():.2f} €")

    # Total consumption, total production
    total_consumption = np.sum(P_load_matrix, axis=0)
    total_production  = np.sum(P_production_matrix, axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(results_df["Time"], total_consumption, 
             label="Total Consumption (kW)", linewidth=2)
    plt.plot(results_df["Time"], results_df["P_discharge"], 
             label="Battery Discharge (kW)", linestyle='dashed', linewidth=2)
    plt.plot(results_df["Time"], results_df["P_grid"], 
             label="Grid Import (kW)", linestyle='dashdot', linewidth=2)
    plt.plot(results_df["Time"], total_production, 
             label="Total Production (kW)", linestyle='dotted', linewidth=2)
    plt.xlabel("Time Step (8H intervals)")
    plt.ylabel("Power (kW)")
    plt.title("Energy Flows Over Time (8-hour Blocks)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "Energy_Flows_Over_Time.png"))
    plt.show()
