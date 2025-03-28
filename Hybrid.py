import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pyswarm import pso
from scipy.optimize import minimize
import time

#%% INITIAL CONFIG
E_max = np.array([2500])
eta_c = np.array([0.95])
eta_d = np.array([0.95])
initial_SoC = E_max * 0.5
C_grid = 0.1

# 8-hour blocks (480 minutes)
AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15  # 32

output_folder = "hybrid_pso_strict_revised"
os.makedirs(output_folder, exist_ok=True)

#%% LOAD DATA
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

# Aggregate into 8-hour blocks
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = consumers_data.shape[0]
P_load_matrix = consumers_data.values.T       # shape: (n_consumers, T)
P_production_matrix = producers_data.values.T # shape: (n_producers, T)

n_consumers = P_load_matrix.shape[0]
n_producers = P_production_matrix.shape[0]

# Example "strict" battery constraints
SoC_min = 0.10 * E_max[0]  # Min 10% SoC
P_c_max = 0.25 * E_max[0]  # Max 25% E_max charging per block
P_d_max = 0.25 * E_max[0]  # Max 25% E_max discharging per block

# Create T-length arrays if needed
P_disch_max_flat = np.tile(E_max, T)
# We can allow a large upper bound for grid (like in the PSO code) OR
# we can do a “per-consumer” approach if you prefer. For clarity, let's just do something large:
P_grid_max = np.full(n_consumers * T, 1e6)

#------------------------------------------------------------------------------
#  Denormalize Utility
#------------------------------------------------------------------------------
def denormalize_vector(x, lower, upper):
    """Converts x in [0,1] to [lower, upper]."""
    return lower + x * (upper - lower)

#------------------------------------------------------------------------------
#  Objective: Just the Grid Cost
#------------------------------------------------------------------------------
def objective(x):
    """
    We remove fallback penalties, unused_production, grid_priority, etc.
    Minimizing: sum(P_grid) * C_grid.
    """
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

    # The objective is simply the total grid cost
    cost_grid = np.sum(P_grid) * C_grid
    return cost_grid

#------------------------------------------------------------------------------
#  Constraint Function (f_ieqcons)
#------------------------------------------------------------------------------
def f_ieqcons(x):
    """
    We define constraints in the form g(x) >= 0, combining:

      1) SoC(t+1) between [SoC_min, E_max]
      2) No over-discharge: P_d(t) <= SoC(t)
      3) Charging limit: sum(P_c(:, t)) <= P_c_max
      4) Discharge limit: P_d(t) <= P_d_max
      5) Production limit: P_c(p, t) <= P_production_matrix(p, t)
      6) Supply–Demand:  battery_discharge + grid - load = 0
         --> implemented via two inequalities to force mismatch in [-eps, +eps].
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

    SoC = np.zeros((1, T+1))
    SoC[0, 0] = initial_SoC[0]

    constraints_list = []

    for t in range(T):

        # (a) sum(P_c) <= P_c_max
        p_c_total = np.sum(P_c[:, t])
        constraints_list.append(P_c_max - p_c_total)  # >= 0 => p_c_total <= P_c_max

        # (b) P_d(t) <= P_d_max
        constraints_list.append(P_d_max - P_d[0, t])

        # Update SoC
        SoC[0, t+1] = SoC[0, t] + eta_c[0]*p_c_total - P_d[0, t]

        # (c) Over-discharge => P_d(t) <= SoC(t)
        constraints_list.append(SoC[0, t] - P_d[0, t])

        # (d) SoC_min <= SoC[t+1] <= E_max
        constraints_list.append(SoC[0, t+1] - SoC_min)   # >= 0 => SoC[t+1] >= SoC_min
        constraints_list.append(E_max[0] - SoC[0, t+1]) # >= 0 => SoC[t+1] <= E_max

        # (e) Production bound => P_c[p,t] <= P_production_matrix[p,t]
        for p in range(n_producers):
            constraints_list.append(P_production_matrix[p, t] - P_c[p, t])

        # (f) Supply–Demand => mismatch in [-epsilon, +epsilon]
        #    mismatch = discharge*eta_d + sum(P_grid[:, t]) - sum(P_load_matrix[:, t])
        supply = P_d[0, t]*eta_d[0] + np.sum(P_grid[:, t])
        demand = np.sum(P_load_matrix[:, t])
        mismatch = supply - demand
        # We enforce: mismatch >= -epsilon => mismatch + epsilon >= 0
        constraints_list.append(mismatch + epsilon)
        # and mismatch <= +epsilon => -mismatch + epsilon >= 0
        constraints_list.append(-mismatch + epsilon)

    return np.array(constraints_list)

#------------------------------------------------------------------------------
# 1) Load an initial guess (seed) from a "simplex" solution
#------------------------------------------------------------------------------
simplex_df = pd.read_csv("simplex_plots/simplex_results.csv")
P_charge_simplex = simplex_df["P_charge"].values   # length T
P_discharge_simplex = simplex_df["P_discharge"].values
P_grid_simplex = simplex_df["P_grid"].values       # length T

# Rebuild them into full dimension:
#   for P_c, we have n_producers * T
#   for P_d, we have T
#   for P_grid, we have n_consumers * T
x_seed_real = np.concatenate([
    np.tile(P_charge_simplex, n_producers),  # naive approach
    P_discharge_simplex,
    np.tile(P_grid_simplex, n_consumers)
])

# We must normalize to [0,1]:
x_upper = np.concatenate([
    P_production_matrix.flatten(),
    P_disch_max_flat,
    P_grid_max
])
x_seed_norm = np.clip(x_seed_real / x_upper, 0, 1)

n_vars = len(x_seed_norm)

#------------------------------------------------------------------------------
# 2) PSO with Strict Constraints
#------------------------------------------------------------------------------
start_pso = time.time()
x_best, fopt = pso(
    objective,
    lb=np.zeros(n_vars),
    ub=np.ones(n_vars),
    f_ieqcons=f_ieqcons,  # Hard constraints via PySwarms
    swarmsize=50,
    maxiter=300,
    minstep=1e-6,
    debug=True,
    x0=x_seed_norm  # optional initial population
)
end_pso = time.time()

print(f"\nPSO completed in {end_pso - start_pso:.2f} seconds.")
print(f"PSO best objective (grid cost) = {fopt:.2f} €")

#------------------------------------------------------------------------------
# 3) Local Refinement (SLSQP)
#------------------------------------------------------------------------------
print("\n🔧 Starting local optimization (SLSQP)...")
start_local = time.time()

res = minimize(
    objective,
    x_best,
    method="SLSQP",
    bounds=[(0, 1)] * n_vars,
    constraints=[
        {'type': 'ineq', 'fun': f_ieqcons}
    ],
    options={"maxiter": 500, "disp": True, "iprint": 2}
)

end_local = time.time()
print("✅ Local optimization completed.")
print(f"⏱️ SLSQP Duration: {end_local - start_local:.2f} seconds")

x_refined = res.x
fopt_refined = res.fun
print(f"Refined objective (grid cost) = {fopt_refined:.2f} €")

#------------------------------------------------------------------------------
# 4) Reconstruct Final Decision Variables
#------------------------------------------------------------------------------
offset1 = n_producers * T
offset2 = offset1 + T
offset3 = offset2 + n_consumers * T

P_c_opt = denormalize_vector(
    x_refined[:offset1],
    0,
    P_production_matrix.flatten()
).reshape(n_producers, T)

P_d_opt = denormalize_vector(
    x_refined[offset1:offset2],
    0,
    P_disch_max_flat
).reshape(1, T)

P_grid_opt = denormalize_vector(
    x_refined[offset2:offset3],
    0,
    P_grid_max
).reshape(n_consumers, T)

# Compute actual SoC with feasible discharge
SoC = np.zeros((1, T + 1))
SoC[0, 0] = initial_SoC[0]
P_d_real = np.zeros_like(P_d_opt)

for t in range(T):
    # Discharge cannot exceed available SoC
    P_d_real[0, t] = min(P_d_opt[0, t], SoC[0, t])
    SoC[0, t+1] = SoC[0, t] + eta_c[0]*np.sum(P_c_opt[:, t]) - P_d_real[0, t]
    SoC[0, t+1] = np.clip(SoC[0, t+1], 0, E_max[0])

P_charge_total    = np.sum(P_c_opt, axis=0)
P_discharge_total = P_d_real[0]
Battery_SoC_total = SoC[0, 1:]
P_grid_total      = np.sum(P_grid_opt, axis=0)

results_df = pd.DataFrame({
    "Time":        np.arange(T),
    "P_grid":      P_grid_total,
    "P_charge":    P_charge_total,
    "P_discharge": P_discharge_total,
    "Battery_SoC": Battery_SoC_total
})
results_df["Cost"] = results_df["P_grid"] * C_grid
results_path = os.path.join(output_folder, "hybrid_refined_results_strict.csv")
results_df.to_csv(results_path, index=False)

#------------------------------------------------------------------------------
# 5) Plot Results
#------------------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Battery_SoC"],
         label="Battery State of Charge (kWh)",
         linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery SoC Over Time (8-hour Blocks)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time_Strict.png"))
plt.show()

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
plt.savefig(os.path.join(output_folder, "Charging_Discharging_Patterns_Strict.png"))
plt.show()

# Consumption vs. Battery vs. Grid vs. Production
total_consumo = np.sum(P_load_matrix, axis=0)
total_producao = np.sum(P_production_matrix, axis=0)

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], total_consumo, label="Total Consumption (kW)",
         linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"],
         label="Battery Discharge (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Import (kW)",
         linestyle='dashdot', linewidth=2)
plt.plot(results_df["Time"], total_producao, label="Total Production (kW)",
         linestyle='dotted', linewidth=2)

plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Power (kW)")
plt.title("Energy Flows Over Time (8-hour Blocks, Strict Constraints)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Energy_Flows_Over_Time_Strict.png"))
plt.show()

#------------------------------------------------------------------------------
# 6) Print Final Results
#------------------------------------------------------------------------------
print(f"\n🔍 Simplex Cost (no penalties): {np.sum(P_grid_simplex) * C_grid:.2f} €")
print(f"✅ Hybrid PSO (initial) Cost: {fopt:.2f} €")
print(f"🔧 Refined Cost (SLSQP) with strict constraints: {fopt_refined:.2f} €")
print(f"   => Final Grid Cost: {results_df['Cost'].sum():.2f} €")
