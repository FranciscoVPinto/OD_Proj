import os
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *

#%%

output_folder = "simplex_plots"
os.makedirs(output_folder, exist_ok=True)

# Set aggregation interval to 480 minutes (8 hours)
AGGREGATION_INTERVAL = 480  # in minutes
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15  # Since original data is in 15 min intervals

# Load data
consumers_file = "Dataset_Consumers.xlsx"
producers_file = "Dataset_Producers.xlsx"

consumers_data = pd.read_excel(consumers_file, header=0)
producers_data = pd.read_excel(producers_file, header=0)

# Ensure dataset length is a multiple of TIME_STEP_RATIO
num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

# Aggregate data to the selected interval
consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = range(len(consumers_data))


T = range(len(consumers_data))

#%%

model = ConcreteModel()
model.T = Set(initialize=T)  

E_max = 1000  # Maximum battery capacity (kWh)
eta_c = 0.95  # Charging efficiency
eta_d = 0.95  # Discharging efficiency
C = {t: 0.1 for t in T}  # Grid energy cost

P_load = {t: consumers_data.iloc[t, :51].sum() for t in T}  
P_production = {t: producers_data.iloc[t, :15].sum() for t in T}  

#%%

model.P_c = Var(T, within=NonNegativeReals)  # Battery charging power
model.P_d = Var(T, within=NonNegativeReals)  # Battery discharging power
model.E = Var(T, within=NonNegativeReals, bounds=(0, E_max))  # Battery energy level
model.P_grid = Var(T, within=NonNegativeReals)  # Grid import power

model.E[0].fix(450)

#%% Constraints

# Battery energy balance over time
def energy_balance_rule(model, t):
    if t == 0:
        return Constraint.Skip 
    return model.E[t] == model.E[t-1] + eta_c * model.P_c[t] - model.P_d[t] / eta_d

model.energy_balance = Constraint(T, rule=energy_balance_rule)

# Battery charging does not exceed available power production
def battery_charging_production_limit_rule(model, t):
    return model.P_c[t] <= P_production[t]

model.battery_charging_production_limit = Constraint(T, rule=battery_charging_production_limit_rule)

# Grid and battery together provide enough power to meet demand
def grid_sufficiency_rule(model, t):
    return model.P_d[t] + model.P_grid[t] >= P_load[t] - P_production[t]

model.grid_sufficiency = Constraint(T, rule=grid_sufficiency_rule)

# Total power supply (grid, battery, production) meets or exceeds demand
def power_balance_rule(model, t):
    return model.P_grid[t] + model.P_d[t] + P_production[t] >= P_load[t]

model.power_balance = Constraint(T, rule=power_balance_rule)

# Battery cannot discharge more energy than it stored in the previous time step
def battery_energy_limit_rule(model, t):
    if t == 0:
        return Constraint.Skip
    return model.P_d[t] <= model.E[t-1]

model.battery_energy_limit = Constraint(T, rule=battery_energy_limit_rule)

# Battery does not discharge more energy than it currently holds
def battery_discharge_requirement_rule(model, t):
    return model.P_d[t] <= model.E[t]

model.battery_discharge_requirement = Constraint(T, rule=battery_discharge_requirement_rule)


#%% Objective Function

model.obj = Objective(expr=sum(C[t] * model.P_grid[t] for t in T), sense=minimize)

#%% Solving

solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

if results.solver.termination_condition != TerminationCondition.optimal:
    print("Solver did not find an optimal solution!")
else:
    print("Solver found an optimal solution!")

#%% Results

results_df = pd.DataFrame({
    "Time": list(T),
    "P_grid": [model.P_grid[t].value if model.P_grid[t].value is not None else 0 for t in T],
    "P_charge": [model.P_c[t].value if model.P_c[t].value is not None else 0 for t in T],
    "P_discharge": [model.P_d[t].value if model.P_d[t].value is not None else 0 for t in T],
    "Battery_SoC": [model.E[t].value if model.E[t].value is not None else 0 for t in T]
})


results_df["Cost"] = results_df["P_grid"] * 0.1  

Total_Cost = results_df["Cost"].sum()
print(f"Total Cost: ${Total_Cost:.2f}")

results_df.to_csv(os.path.join(output_folder, "meta_results.csv"), index=False)

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Cost"], label="Cost Over Time ($)", linewidth=2, color='r')
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Cost ($)")
plt.title("Cost Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Cost_Over_Time.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="Battery State of Charge (SoC) (kWh)", linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery SoC Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time.png"))
plt.show()
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Power Import (kW)", linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Grid Power Import (kW)")
plt.title("Grid Power Import Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Grid_Power_Import.png"))
plt.show()
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_charge"], label="Battery Charging (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharging (kW)", linestyle='solid', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Power (kW)")
plt.title("Charging and Discharging Patterns")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Charging_Discharging_Patterns.png"))
plt.show()
plt.close()
