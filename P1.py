import os
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *

output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

consumers_file = "Dataset_Consumers.xlsx"
producers_file = "Dataset_Producers.xlsx"

consumers_data = pd.read_excel(consumers_file, header=0)
producers_data = pd.read_excel(producers_file, header=0)

T = range(len(consumers_data))

model = ConcreteModel()
model.T = Set(initialize=T)

E_max = 200  
eta_c = 0.95  
eta_d = 0.95  
C = {t: 0.1 for t in T}  

P_load = {t: consumers_data.iloc[t, :51].sum() for t in T}
P_production = {t: producers_data.iloc[t, 1:].sum() for t in T}

model.P_c = Var(T, within=NonNegativeReals)  
model.P_d = Var(T, within=NonNegativeReals)  
model.E = Var(T, within=NonNegativeReals, bounds=(0, E_max))  
model.P_grid = Var(T, within=NonNegativeReals, bounds=(0, None))  

model.E[0].fix(0)

def energy_balance_rule(model, t):
    if t == 0:
        return Constraint.Skip  
    return model.E[t] == model.E[t-1] + eta_c * model.P_c[t] - model.P_d[t] / eta_d

model.energy_balance = Constraint(T, rule=energy_balance_rule)

def power_balance_rule(model, t):
    return model.P_grid[t] + P_production[t] + model.P_d[t] == P_load[t]

model.power_balance = Constraint(T, rule=power_balance_rule)

def battery_discharge_limit_rule(model, t):
    if t == 0:
        return Constraint.Skip
    return model.P_d[t] <= P_load[t] - P_production[t]

model.battery_discharge_limit = Constraint(T, rule=battery_discharge_limit_rule)

def battery_energy_limit_rule(model, t):
    if t == 0:
        return Constraint.Skip
    return model.P_d[t] <= model.E[t-1]

model.battery_energy_limit = Constraint(T, rule=battery_energy_limit_rule)

def battery_charging_rule(model, t):
    return model.P_c[t] <= P_production[t]  

model.battery_charging = Constraint(T, rule=battery_charging_rule)

def grid_usage_rule(model, t):
    return model.P_grid[t] >= P_load[t] - (P_production[t] + model.P_d[t])

model.grid_usage = Constraint(T, rule=grid_usage_rule)

model.P_d.setlb(0)  

model.obj = Objective(expr=sum(C[t] * model.P_grid[t] for t in T), sense=minimize)

solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

if results.solver.termination_condition != TerminationCondition.optimal:
    print("⚠️ Solver did not find an optimal solution!")
else:
    print("✅ Solver found an optimal solution!")

time_range = list(T)
P_grid_values = [model.P_grid[t].value if model.P_grid[t].value is not None else 0 for t in time_range]
P_c_values = [model.P_c[t].value if model.P_c[t].value is not None else 0 for t in time_range]
P_d_values = [model.P_d[t].value if model.P_d[t].value is not None else 0 for t in time_range]
E_values = [model.E[t].value if model.E[t].value is not None else 0 for t in time_range]

print("🔎 Initial Conditions")
for t in range(min(5, len(T))):
    print(f"\n🔍 Debugging Constraints at Time {t}")
    print(f"🔹 P_load[{t}] = {P_load[t]}")
    print(f"🔹 P_production[{t}] = {P_production[t]}")
    print(f"🔹 P_d[{t}] must be ≤ {P_load[t] - P_production[t]}")
    print(f"🔹 P_grid[{t}] must be ≥ {P_load[t]} - (P_production[{t}] + P_d[{t}])")
    print(f"✅ Solver Results: P_grid[{t}] = {P_grid_values[t]} kW")

plt.figure(figsize=(14, 6))
plt.plot(time_range, E_values, label="State of Charge (SoC) - Battery Energy (kWh)", linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery SoC Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time.png"))
plt.close()

print(f"✅ Optimization complete. Results are available for further analysis.")
