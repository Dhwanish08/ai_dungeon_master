# src/eda/monster_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================
# 1. Setup
# =============================
df = pd.read_csv("data/processed/Dd5e_monsters_clean.csv")
df.columns = df.columns.str.strip().str.lower()

# Ensure reports/figures exists
figures_path = "reports/figures"
os.makedirs(figures_path, exist_ok=True)

# =============================
# 2. Alignment distribution
# =============================
plt.figure(figsize=(10, 6))
df["alignment"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Monster Alignment Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{figures_path}/alignment_distribution.png")
plt.close()

# =============================
# 3. Monster size distribution
# =============================
plt.figure(figsize=(8, 6))
df["size"].value_counts().plot(kind="bar", color="salmon", edgecolor="black")
plt.title("Monster Size Distribution")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{figures_path}/size_distribution.png")
plt.close()

# =============================
# 4. Challenge Rating histogram
# =============================
plt.figure(figsize=(10, 6))
df["challenge_rating"].astype(str).value_counts().sort_index().plot(
    kind="bar", color="lightgreen", edgecolor="black"
)
plt.title("Challenge Rating Distribution")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figures_path}/challenge_rating_distribution.png")
plt.close()

# =============================
# 5. Armor Class vs Challenge Rating
# =============================
plt.figure(figsize=(8, 6))
plt.scatter(df["armor_class"], df["challenge_rating"].str.extract(r"(\d+)").astype(float), alpha=0.6, c="purple")
plt.xlabel("Armor Class")
plt.ylabel("Challenge Rating")
plt.title("Armor Class vs Challenge Rating")
plt.tight_layout()
plt.savefig(f"{figures_path}/ac_vs_cr.png")
plt.close()

# =============================
# 6. Hit Points vs Challenge Rating
# =============================
plt.figure(figsize=(8, 6))
plt.scatter(df["hit_points"], df["challenge_rating"].str.extract(r"(\d+)").astype(float), alpha=0.6, c="orange")
plt.xscale("log")
plt.xlabel("Hit Points (log scale)")
plt.ylabel("Challenge Rating")
plt.title("Hit Points vs Challenge Rating")
plt.tight_layout()
plt.savefig(f"{figures_path}/hp_vs_cr.png")
plt.close()

print(f"EDA plots saved to: {figures_path}")
