import os
import pandas as pd
import matplotlib.pyplot as plt

# Create a directory for plots if it doesn't exist
output_folder = "plots_init"
os.makedirs(output_folder, exist_ok=True)

# Load datasets, skipping the first row
consumers_file = "Dataset_Consumers.xlsx"
producers_file = "Dataset_Producers.xlsx"

# Read consumer and producer data, skipping the first row
consumers_data = pd.read_excel(consumers_file, header=0, skiprows=1)
producers_data = pd.read_excel(producers_file, header=0, skiprows=1)

# Extract time steps
time_range = range(len(consumers_data))

# Sum up total energy consumption and production for each time step
P_load_values = [consumers_data.iloc[t, 1:].sum() for t in time_range]
P_production_values = [producers_data.iloc[t, 1:].sum() for t in time_range]

# ✅ Plot Production Before Optimization
plt.figure(figsize=(14, 6))
plt.plot(time_range, P_production_values, label="Total Energy Produced (kWh)", color='green', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Energy (kWh)")
plt.title("Energy Production Over Time (Before Optimization)")
plt.legend()
plt.grid()
plot_production_path = os.path.join(output_folder, "Energy_Production_Before_Optimization.png")
plt.savefig(plot_production_path)
plt.close()

# ✅ Plot Consumption Before Optimization
plt.figure(figsize=(14, 6))
plt.plot(time_range, P_load_values, label="Total Energy Consumed (kWh)", color='red', linewidth=2)
plt.xlabel("Time Step (15 min intervals)")
plt.ylabel("Energy (kWh)")
plt.title("Energy Consumption Over Time (Before Optimization)")
plt.legend()
plt.grid()
plot_consumption_path = os.path.join(output_folder, "Energy_Consumption_Before_Optimization.png")
plt.savefig(plot_consumption_path)
plt.close()

print(f"✅ Production plot saved: {plot_production_path}")
print(f"✅ Consumption plot saved: {plot_consumption_path}")
