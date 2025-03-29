import os
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *

#%% Parâmetros fixos
N_batteries = 1
E_max = [2500]
eta_c = [0.95]
eta_d = [0.95]
initial_SoC = [E_max[0] * 0.5]
C_grid = 0.1

AGGREGATION_INTERVAL = 480
TIME_STEP_RATIO = AGGREGATION_INTERVAL // 15

output_folder = "simplex_1p1c_plots"
os.makedirs(output_folder, exist_ok=True)

#%% Carregar dados
consumers_file = "Top_2_Months_Consumers.xlsx"
producers_file = "Top_2_Months_Production.xlsx"

consumers_data = pd.read_excel(consumers_file, skiprows=1)
producers_data = pd.read_excel(producers_file, skiprows=1)

num_rows = (len(consumers_data) // TIME_STEP_RATIO) * TIME_STEP_RATIO
consumers_data = consumers_data.iloc[:num_rows]
producers_data = producers_data.iloc[:num_rows]

consumers_data = consumers_data.groupby(consumers_data.index // TIME_STEP_RATIO).sum()
producers_data = producers_data.groupby(producers_data.index // TIME_STEP_RATIO).sum()

T = range(len(consumers_data))
P_load = consumers_data.sum(axis=1).to_dict()
P_production = producers_data.sum(axis=1).to_dict()

#%% Modelo Pyomo
model = ConcreteModel()
model.T = Set(initialize=T)
model.B = Set(initialize=range(N_batteries))

model.P_c = Var(model.B, model.T, within=NonNegativeReals)
model.P_d = Var(model.B, model.T, within=NonNegativeReals)
model.SoC = Var(model.B, model.T, within=NonNegativeReals, bounds=(0, E_max[0]))
model.P_grid = Var(model.T, within=NonNegativeReals)

model.SoC[0, 0].fix(initial_SoC[0])

#%% Restrições
def soc_balance_rule(model, i, t):
    if t == 0:
        return Constraint.Skip
    return model.SoC[i, t] == model.SoC[i, t-1] + eta_c[i] * model.P_c[i, t] - model.P_d[i, t] / eta_d[i]
model.soc_balance = Constraint(model.B, model.T, rule=soc_balance_rule)

def charging_limit_rule(model, i, t):
    return model.P_c[i, t] <= P_production[t]
model.charging_limit = Constraint(model.B, model.T, rule=charging_limit_rule)

def supply_rule(model, t):
    return sum(model.P_d[i, t] for i in model.B) + model.P_grid[t] >= P_load[t] - P_production[t]
model.supply_rule = Constraint(model.T, rule=supply_rule)

def discharge_limit_rule(model, i, t):
    return model.P_d[i, t] <= model.SoC[i, t]
model.discharge_limit = Constraint(model.B, model.T, rule=discharge_limit_rule)

#%% Função Objetivo
model.obj = Objective(expr=sum(C_grid * model.P_grid[t] for t in model.T), sense=minimize)

#%% Otimização
solver = SolverFactory('glpk')
results = solver.solve(model, tee=False)

#%% Resultados
results_df = pd.DataFrame({
    "Time": list(model.T),
    "P_grid": [model.P_grid[t]() for t in model.T],
    "P_charge": [model.P_c[0, t]() for t in model.T],
    "P_discharge": [model.P_d[0, t]() for t in model.T],
    "Battery_SoC": [model.SoC[0, t]() for t in model.T],
})
results_df["Cost"] = results_df["P_grid"] * C_grid
results_df.to_csv(os.path.join(output_folder, "simplex_results.csv"), index=False)

#%% Gráficos

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["Battery_SoC"], label="Battery State of Charge (SoC) (kWh)", linestyle='dashed', linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Battery Energy (kWh)")
plt.title("Battery SoC Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Battery_SoC_Over_Time.png"))
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], results_df["P_charge"], label="Battery Charging (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharging (kW)", linestyle='solid', linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Power (kW)")
plt.title("Charging and Discharging Patterns")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Charging_Discharging_Patterns.png"))
plt.show()

print(f"Custo Total (SIMPLEX): {results_df['Cost'].sum():.2f} €")

#%% Gráfico: Consumo, Descarga da Bateria, Importação da Rede e Produção
total_consumo = list(P_load.values())
total_producao = list(P_production.values())

plt.figure(figsize=(14, 6))
plt.plot(results_df["Time"], total_consumo, label="Total Consumption (kW)", linewidth=2)
plt.plot(results_df["Time"], results_df["P_discharge"], label="Battery Discharge (kW)", linestyle='dashed', linewidth=2)
plt.plot(results_df["Time"], results_df["P_grid"], label="Grid Import (kW)", linestyle='dashdot', linewidth=2)
plt.plot(results_df["Time"], total_producao, label="Total Production (kW)", linestyle='dotted', linewidth=2)

plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Power (kW)")
plt.title("Energy Flows Over Time")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_folder, "Energy_Flows_Over_Time.png"))
plt.show()


