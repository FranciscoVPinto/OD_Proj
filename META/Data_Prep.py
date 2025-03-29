import pandas as pd
import matplotlib.pyplot as plt

producers_file = "Dataset_Producers.xlsx"
consumers_file = "Dataset_Consumers.xlsx"

xls_prod = pd.ExcelFile(producers_file)
xls_cons = pd.ExcelFile(consumers_file)

df_prod = xls_prod.parse("Total Producers")
df_cons = xls_cons.parse("Total Consumers")

start_time = pd.Timestamp("2024-01-01 00:00")
time_index = pd.date_range(start=start_time, periods=df_prod.shape[0], freq='15min')

df_prod.columns = [f'Producer_{i+1}' for i in range(df_prod.shape[1])]
df_temp = df_prod.copy()
df_temp['Timestamp'] = time_index
df_temp['Total_Production'] = df_temp[[f'Producer_{i+1}' for i in range(15)]].sum(axis=1)
df_temp['Month'] = df_temp['Timestamp'].dt.month

monthly_production = df_temp.groupby('Month')['Total_Production'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.bar(monthly_production['Month'], monthly_production['Total_Production'])
plt.xlabel("Mês")
plt.ylabel("Produção Total (kWh)")
plt.title("Produção Mensal Total dos Produtores")
plt.xticks(range(1, 13))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("producao_mensal.png")

top_two_months = monthly_production.sort_values(by='Total_Production', ascending=False).head(2)
selected_months = top_two_months['Month'].tolist()
filtered_temp = df_temp[df_temp['Month'].isin(selected_months)]

filtered_clean = filtered_temp[[f'Producer_{i+1}' for i in range(15)]]
filtered_clean.to_excel("Top_2_Months_Production.xlsx", index=False)

df_cons_top_months = df_cons.iloc[:len(filtered_temp), :].copy()
df_cons_top_months.columns = [f'Consumer_{i+1}' for i in range(df_cons_top_months.shape[1])]
df_cons_top_months.to_excel("Top_2_Months_Consumers.xlsx", index=False)
