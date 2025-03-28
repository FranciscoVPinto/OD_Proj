import pandas as pd
import matplotlib.pyplot as plt
import os


# Carregar resultados
simplex = pd.read_csv("simplex_plots/simplex_results.csv")
de = pd.read_csv("de_plots/de_results.csv")
pso = pd.read_csv("pso_plots/pso_results.csv")

output_folder = "Final_Results"
os.makedirs(output_folder, exist_ok=True)

# Gráfico 1: Custo acumulado
plt.figure(figsize=(14, 5))
plt.plot(simplex["Time"], simplex["Cost"].cumsum(), label="Simplex", linewidth=2)
plt.plot(de["Time"], de["Cost"].cumsum(), label="Differential Evolution", linewidth=2)
plt.plot(pso["Time"], pso["Cost"].cumsum(), label="PSO", linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Cumulative Cost (€)")
plt.title("Custo acumulado ao longo do tempo")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"comparative_costs.png"))
plt.show()

# Gráfico 2: Energia importada da rede
plt.figure(figsize=(14, 5))
plt.plot(simplex["Time"], simplex["P_grid"], label="Simplex", linewidth=2)
plt.plot(de["Time"], de["P_grid"], label="DE", linewidth=2)
plt.plot(pso["Time"], pso["P_grid"], label="PSO", linewidth=2)
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Grid Import (kW)")
plt.title("Energia Importada da Rede")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"comparative_grid_power.png"))
plt.show()

# Gráfico 3: Estado de carga total (SoC)
plt.figure(figsize=(14, 5))
plt.plot(simplex["Time"], simplex["Battery_SoC"], label="Simplex", linestyle='--')
plt.plot(de["Time"], de["Battery_SoC"], label="DE", linestyle='-')
plt.plot(pso["Time"], pso["Battery_SoC"], label="PSO", linestyle='-.')
plt.xlabel("Time Step (8H intervals)")
plt.ylabel("Total Battery SoC (kWh)")
plt.title("Estado de Carga das Baterias")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"comparative_soc.png"))
plt.show()
