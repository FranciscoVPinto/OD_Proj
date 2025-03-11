from pyomo.environ import *
import pandas as pd

# Load dataset from CSV
# The dataset contains columns: 'demand', 'renewable_supply', 'grid_price'
data = pd.read_csv("battery_data.csv")

demand = data["demand"].values  # Energy demand at each time step
renewable_supply = data["renewable_supply"].values  # Available renewable energy at each time step
grid_price = data["grid_price"].values  # Cost of grid energy per unit

# Create a Pyomo optimization model
model = ConcreteModel()

# Sets
T = len(demand)  # Number of time periods
N = 3   # Number of batteries
model.T = RangeSet(0, T-1)  # Set of time periods
model.N = RangeSet(0, N-1)  # Set of batteries

# Battery parameters (constant values, can be loaded from dataset if needed)
capacity = [50, 60, 40]  # Maximum energy capacity of each battery
charge_limit = [10, 12, 8]  # Maximum charge rate of each battery
initial_energy = [20, 30, 15]  # Initial energy stored in each battery
discharge_limit = charge_limit  # Maximum discharge rate, assumed symmetric

# Decision Variables
# Amount of power charged, discharged, and stored in each battery
model.charge = Var(model.N, model.T, within=NonNegativeReals)
model.discharge = Var(model.N, model.T, within=NonNegativeReals)
model.energy = Var(model.N, model.T, within=NonNegativeReals)
# Energy purchased from the grid
model.grid_energy = Var(model.T, within=NonNegativeReals)

# Objective function: Minimize total grid energy cost
def obj_rule(model):
    return sum(grid_price[t] * model.grid_energy[t] for t in model.T)
model.objective = Objective(rule=obj_rule, sense=minimize)

# Constraints

# Energy balance: Demand must be met by grid, battery discharge, or renewables
def energy_balance_rule(model, t):
    return model.grid_energy[t] + sum(model.discharge[i, t] for i in model.N) + renewable_supply[t] >= demand[t]
model.energy_balance = Constraint(model.T, rule=energy_balance_rule)

# Battery energy state transition equation
def battery_transition_rule(model, i, t):
    if t == 0:
        return model.energy[i, t] == initial_energy[i] + model.charge[i, t] - model.discharge[i, t]
    else:
        return model.energy[i, t] == model.energy[i, t-1] + model.charge[i, t] - model.discharge[i, t]
model.battery_transition = Constraint(model.N, model.T, rule=battery_transition_rule)

# Battery capacity constraint
def capacity_rule(model, i, t):
    return model.energy[i, t] <= capacity[i]
model.capacity = Constraint(model.N, model.T, rule=capacity_rule)

# Battery charge limit constraint
def charge_limit_rule(model, i, t):
    return model.charge[i, t] <= charge_limit[i]
model.charge_limit = Constraint(model.N, model.T, rule=charge_limit_rule)

# Battery discharge limit constraint
def discharge_limit_rule(model, i, t):
    return model.discharge[i, t] <= discharge_limit[i]
model.discharge_limit = Constraint(model.N, model.T, rule=discharge_limit_rule)

# Solve the model using the Simplex method (GLPK solver)
solver = SolverFactory('glpk')
solver.solve(model)

# Print results
for t in model.T:
    print(f"Time {t}: Grid Energy Used = {model.grid_energy[t].value:.2f} kWh")
    for i in model.N:
        print(f"  Battery {i}: Charge = {model.charge[i, t].value:.2f} kWh, Discharge = {model.discharge[i, t].value:.2f} kWh, Energy Level = {model.energy[i, t].value:.2f} kWh")
