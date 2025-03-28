import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyswarm import pso
import os

#%% Fixed Parameters
N_batteries = 1
E_max = np.array([2500])   # Battery capacity (kWh)
eta_c = np.array([0.95])   # Charging efficiency
eta_d = np.array([0.95])   # Discharging efficiency
initial_SoC = E_max * 0.5  # 50% initial SoC
C_grid = 0.1               # Grid cost (€/kWh)

# 4-hour blocks instead of 8:
AGGREGATION_INTERVAL = 480                 # 4 hours in minutes
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15  # 16

output_folder = "pso_plots"
os.makedirs(output_folder, exist_ok=True)

#%% Load Data
consumers_data = pd.read_excel("Top_2_Months_Consumers.xlsx", skiprows=1)
producers_data = pd.read_excel("Top_2_Months_Production.xlsx", skiprows=1)

# Truncate leftover rows so it's divisible by TIME_STEP_RATIO
num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

# Aggregate into 4-hour blocks
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = consumers_data.shape[0]
P_load_matrix = consumers_data.values.T       # shape: (n_consumers, T)
P_production_matrix = producers_data.values.T # shape: (n_producers, T)

n_consumers = P_load_matrix.shape[0]
n_producers = P_production_matrix.shape[0]

#%% No Grid Limitation: set a large upper bound
# We'll let each consumer/time step import a big number (e.g., 1e6).
# Flatten into shape (n_consumers*T,)
P_grid_max = np.full(n_consumers * T, 1e6)

# Discharge bounds: let the battery discharge up to entire E_max each block
P_disch_max_flat = np.tile(E_max, T)

#%% Denormalization Function
def denormalize_vector(x, lower, upper):
    """
    Convert x in [0,1] to [lower,upper].
    """
    return lower + x * (upper - lower)

#%% Objective Function
def objective(x):
    """
    Minimize total grid cost + penalties for SoC issues.
    """
    penalty_weights = {
        "max_power": 200,
        "SoC_violation": 200,
        "over_discharge": 200,
    }

    # Offsets
    offset1 = n_producers * T             # length of P_c
    offset2 = offset1 + T                 # length of P_d
    offset3 = offset2 + n_consumers * T   # length of P_grid

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

    penalty = 0.0
    SoC = np.zeros((1, T + 1))
    SoC[0, 0] = initial_SoC[0]

    # No minimum SoC: SoC_min=0
    # No max discharge limit: P_d_max = E_max[0]
    SoC_min = 0.0
    P_c_max = E_max[0]  # let it charge up to full capacity if it wants
    P_d_max = E_max[0]  # let it discharge entire capacity

    for t in range(T):
        # 1) Enforce max charging
        P_c_total = np.sum(P_c[:, t])
        if P_c_total > P_c_max:
            excess = P_c_total - P_c_max
            penalty += penalty_weights["max_power"] * excess
            if P_c_total > 0:
                scale = P_c_max / P_c_total
                P_c[:, t] *= scale
            P_c_total = P_c_max

        # 2) Enforce max discharging
        if P_d[0, t] > P_d_max:
            excess = P_d[0, t] - P_d_max
            penalty += penalty_weights["max_power"] * excess
            P_d[0, t] = P_d_max

        # 3) Over-discharge check
        if P_d[0, t] > SoC[0, t]:
            over_discharge_amt = P_d[0, t] - SoC[0, t]
            penalty += penalty_weights["over_discharge"] * over_discharge_amt
            P_d[0, t] = SoC[0, t]

        # 4) Update SoC
        SoC[0, t+1] = SoC[0, t] + (eta_c[0] * P_c_total) - P_d[0, t]

        # 5) SoC bounds
        if SoC[0, t+1] < SoC_min:
            diff = SoC_min - SoC[0, t+1]
            penalty += penalty_weights["SoC_violation"] * diff
            SoC[0, t+1] = SoC_min
        elif SoC[0, t+1] > E_max[0]:
            diff = SoC[0, t+1] - E_max[0]
            penalty += penalty_weights["SoC_violation"] * diff
            SoC[0, t+1] = E_max[0]

        # 6) Production bound
        for p in range(n_producers):
            if P_c[p, t] > P_production_matrix[p, t]:
                penalty += penalty_weights["max_power"] * (
                    P_c[p, t] - P_production_matrix[p, t]
                )

    # 7) Grid cost
    grid_cost = np.sum(P_grid) * C_grid
    return grid_cost + penalty

#%% Hard Constraints: supply == demand
def f_ieqcons(x):
    """
    battery discharge + grid = load, within a tolerance epsilon=1e-4
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

    con_vals = np.zeros(2*T)
    for t in range(T):
        battery_energy_t = P_d[0, t] * eta_d[0]
        grid_energy_t = np.sum(P_grid[:, t])
        total_demand_t = np.sum(P_load_matrix[:, t])

        diff = (battery_energy_t + grid_energy_t) - total_demand_t

        # Two inequalities to represent |diff| <= epsilon
        con_vals[t]   = diff + epsilon    # must be >= 0 => diff >= -epsilon
        con_vals[t+T] = -diff + epsilon   # must be >= 0 => diff <= epsilon

    return con_vals

#%% PSO Setup
n_vars = n_producers * T + T + n_consumers * T
lb = np.zeros(n_vars)
ub = np.ones(n_vars)

# Increase swarmsize & maxiter for better chance of feasibility
x_opt, fopt = pso(
    objective,
    lb,
    ub,
    f_ieqcons=f_ieqcons,
    swarmsize=500,    
    maxiter=2000,     
    minstep=1e-6,
    debug=True
)

#%% Process Results
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

SoC = np.zeros((1, T + 1))
SoC[0, 0] = initial_SoC[0]
P_d_real = np.zeros_like(P_d_opt)

for t in range(T):
    # actual feasible discharge can't exceed SoC
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
results_df.to_csv(os.path.join(output_folder, "pso_results_4h.csv"), index=False)

#%% Check Feasibility
violations = []
for t in range(T):
    battery_energy = P_discharge_total[t] * eta_d[0]
    grid_energy = P_grid_total[t]
    total_demand = np.sum(P_load_matrix[:, t])
    diff = (battery_energy + grid_energy) - total_demand

    # Hard equality => diff within ±some small tolerance
    if abs(diff) > 1e-3:
        violations.append(f"❌ Time {t}: supply != demand (diff={diff:.4f} kWh)")

    # SoC checks
    if SoC[0, t] < -1e-6:
        violations.append(f"❌ Time {t}: Negative SoC ({SoC[0, t]:.4f} kWh)")
    if SoC[0, t] > E_max[0] + 1e-6:
        violations.append(f"❌ Time {t}: SoC above max ({SoC[0, t]:.4f} > {E_max[0]})")

if violations:
    print("\n⚠️ Violations found in the final solution:")
    for v in violations:
        print(v)
else:
    print("\n✅ All constraints satisfied: grid + discharge == consumption, SoC within bounds.")

#%% Plotting

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="Battery SoC (kWh)",
         linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (4H intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery State of Charge Over Time (8-hour Blocks)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time_4h.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_charge"], label="Battery Charging (kW)",
         linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharging (kW)",
         linestyle='solid', linewidth=2)
plt.xlabel("Time Step (4H intervals)")
plt.ylabel("Power (kW)")
plt.title("Charging and Discharging Patterns (8-hour Blocks)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Charging_Discharging_Patterns.png"))
plt.show()

print(f"Grid cost (PSO): {results_df['Cost'].sum():.2f} €")
print(f"Total objective (with penalties): {fopt:.2f} €")

#%% Energy Flow Plot
total_consumption = np.sum(P_load_matrix, axis=0)
total_production = np.sum(P_production_matrix, axis=0)

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], total_consumption, label="Total Consumption (kW)", linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharge (kW)",
         linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Import (kW)",
         linestyle='dashdot', linewidth=2)
plt.plot(results_df["Time"], total_production, label="Total Production (kW)",
         linestyle='dotted', linewidth=2)

plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Power (kW)")
plt.title("Energy Flows Over Time (4-hour Blocks)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Energy_Flows_Over_Time_4h.png"))
plt.show()
