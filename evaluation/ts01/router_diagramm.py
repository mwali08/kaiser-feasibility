import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("router_test_summary.csv")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Genauigkeit
sns.barplot(x="Modell", y="Genauigkeit (%)", data=df, ax=axes[0], palette="viridis")
axes[0].axhline(90, color='red', linestyle='--', label='KO-Kriterium (90%)')
axes[0].set_title("Klassifikationsgenauigkeit")
axes[0].legend()

# 2. Latenz
df_melted = df.melt(id_vars=["Modell"], value_vars=["Latenz Median (ms)", "Latenz P95 (ms)"], var_name="Metrik", value_name="Latenz (ms)")
sns.barplot(x="Modell", y="Latenz (ms)", hue="Metrik", data=df_melted, ax=axes[1], palette="magma")
axes[1].set_title("Routing-Latenz")

# 3. Format & FPR
df_melted2 = df.melt(id_vars=["Modell"], value_vars=["Gültiges Format (%)", "Falsch-Positiv-Rate OOS (%)"], var_name="Metrik", value_name="Prozent (%)")
sns.barplot(x="Modell", y="Prozent (%)", hue="Metrik", data=df_melted2, ax=axes[2], palette="coolwarm")
axes[2].set_title("Format-Treue & Falsch-Positive")

plt.tight_layout()
plt.savefig("router_vergleich.png", dpi=300)
plt.show()