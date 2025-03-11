from pyomo.environ import *

# Define model
model = ConcreteModel()

# Sets
T = range(24)  # Time periods (e.g., 24 hours)
model.T = Set(initialize=T)

# Parameters
E_max = 100  # Maximum battery capacity (kWh)
P_c_max = 20  # Max charging power (kW)
P_d_max = 20  # Max discharging power (kW)
eta_c = 0.95  # Charging efficiency
eta_d = 0.95  # Discharging efficiency
C = {t: 0.1 for t in T}  # Electricity price per kWh
P_load = {t: 10 for t in T}  # Load demand (kW)

# Decision Variables
model.P_c = Var(T, within=NonNegativeReals, bounds=(0, P_c_max))  # Charging power
model.P_d = Var(T, within=NonNegativeReals, bounds=(0, P_d_max))  # Discharging power
model.E = Var(T, within=NonNegativeReals, bounds=(0, E_max))  # Battery energy level
model.P_grid = Var(T, within=NonNegativeReals)  # Power from grid

# Objective: Minimize energy cost
model.obj = Objective(expr=sum(C[t] * model.P_grid[t] for t in T), sense=minimize)

# Constraints

def energy_balance_rule(model, t):
    if t == 0:
        return model.E[t] == 50 + eta_c * model.P_c[t] - model.P_d[t] / eta_d  # Initial SoC = 50 kWh
    return model.E[t] == model.E[t-1] + eta_c * model.P_c[t] - model.P_d[t] / eta_d

model.energy_balance = Constraint(T, rule=energy_balance_rule)

model.battery_limits = ConstraintList()
for t in T:
    model.battery_limits.add(0 <= model.E[t] <= E_max)

model.power_balance = ConstraintList()
for t in T:
    model.power_balance.add(model.P_c[t] + model.P_grid[t] == model.P_d[t] + P_load[t])

# Solve the model
solver = SolverFactory('glpk')
solver.solve(model)

# Print results
for t in T:
    print(f"Hour {t}: P_grid = {model.P_grid[t]()} kW, P_c = {model.P_c[t]()} kW, P_d = {model.P_d[t]()} kW, E = {model.E[t]()} kWh")
