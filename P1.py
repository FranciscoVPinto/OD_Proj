import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *

# Load datasets
consumers_file = "Dataset_Consumers.xlsx"
producers_file = "Dataset_Producers.xlsx"

# Read consumer and producer data
consumers_data = pd.read_excel(consumers_file, header=None)
producers_data = pd.read_excel(producers_file, header=None)

# Number of time steps (35117 intervals of 15 minutes)
T = range(len(consumers_data))

# Define model
model = ConcreteModel()
model.T = Set(initialize=T)

# Parameters
E_max = 100  # Maximum battery capacity (kWh)
P_c_max = 20  # Max charging power (kW)
P_d_max = 20  # Max discharging power (kW)
eta_c = 0.95  # Charging efficiency
eta_d = 0.95  # Discharging efficiency
C = {t: 0.1 for t in T}  # Electricity price per kWh

# Calculate total load and production per time step
P_load = {t: consumers_data.iloc[t, 1:].sum() for t in T}  # Sum across all consumers
P_production = {t: producers_data.iloc[t, 1:].sum() for t in T}  # Sum across all producers

# Decision Variables
model.P_c = Var(T, within=NonNegativeReals, bounds=(0, P_c_max))  # Charging power
model.P_d = Var(T, within=NonNegativeReals, bounds=(0, P_d_max))  # Discharging power
model.E = Var(T, within=NonNegativeReals)  # Battery energy level
model.P_grid = Var(T, within=NonNegativeReals)  # Power from grid

# Objective: Minimize energy cost
model.obj = Objective(expr=sum(C[t] * model.P_grid[t] for t in T), sense=minimize)

# Constraints

def energy_balance_rule(model, t):
    if t == 0:
        return model.E[t] == 50 + eta_c * model.P_c[t] - model.P_d[t] / eta_d  # Initial SoC = 50 kWh
    return model.E[t] == model.E[t-1] + eta_c * model.P_c[t] - model.P_d[t] / eta_d

model.energy_balance = Constraint(T, rule=energy_balance_rule)

# Correct battery state of charge constraints
model.battery_min = Constraint(T, rule=lambda model, t: model.E[t] >= 0)
model.battery_max = Constraint(T, rule=lambda model, t: model.E[t] <= E_max)

# Power balance constraint with real-time production and consumption
model.power_balance = ConstraintList()
for t in T:
    model.power_balance.add(model.P_c[t] + model.P_grid[t] + P_production[t] == model.P_d[t] + P_load[t])

# Solve the model using the Simplex method
solver = SolverFactory('glpk')
solver.solve(model)

# Extract results for plotting
time_range = list(T)  # Limit to first 100 time steps
P_grid_values = [model.P_grid[t]() for t in time_range]
P_c_values = [model.P_c[t]() for t in time_range]
P_d_values = [model.P_d[t]() for t in time_range]
E_values = [model.E[t]() for t in time_range]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_range, P_grid_values, label="Power from Grid (kW)")
plt.plot(time_range, P_c_values, label="Charging Power (kW)")
plt.plot(time_range, P_d_values, label="Discharging Power (kW)")
plt.plot(time_range, E_values, label="Battery Energy (kWh)", linestyle='dashed')
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Power/Energy")
plt.title("Battery Dispatch Optimization Results")
plt.legend()
plt.grid()
plt.show()
x