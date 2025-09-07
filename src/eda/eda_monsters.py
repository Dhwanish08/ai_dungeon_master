# src/eda/eda_monsters.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================
# Setup
# =============================
# Ensure figures directory exists
figures_path = "reports/figures"
os.makedirs(figures_path, exist_ok=True)

# Load data
df = pd.read_csv("data/processed/Dd5e_monsters_clean.csv")
df.columns = df.columns.str.strip().str.lower()

# =============================
# Helper function to save plots
# =============================
def save_plot(fig, filename):
    fig.savefig(os.path.join(figures_path, filename), bbox_inches="tight")
    plt.close(fig)

# =============================
# 1. Alignment Distribution
# =============================
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(y="alignment", data=df, order=df["alignment"].value_counts().index, palette="viridis", ax=ax)
ax.set_title("Number of Monsters per Alignment")
ax.set_xlabel("Count")
ax.set_ylabel("Alignment")
save_plot(fig, "alignment_distribution.png")

# =============================
# 2. Size Distribution
# =============================
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="size", data=df, order=df["size"].value_counts().index, palette="plasma", ax=ax)
ax.set_title("Distribution of Monster Sizes")
ax.set_xlabel("Size")
ax.set_ylabel("Count")
save_plot(fig, "size_distribution.png")

# =============================
# 3. Armor Class by Alignment
# =============================
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x="alignment", y="armor_class", data=df, palette="coolwarm", ax=ax)
ax.set_title("Armor Class Distribution by Alignment")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
save_plot(fig, "armor_class_by_alignment.png")

# =============================
# 4. Hit Points by Size
# =============================
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="size", y="hit_points", data=df, palette="magma", ax=ax)
ax.set_title("Hit Points Distribution by Size")
save_plot(fig, "hit_points_by_size.png")

# =============================
# 5. Challenge Rating Distribution
# =============================
df["challenge_rating_clean"] = df["challenge_rating"].str.extract(r"([\d/]+)").fillna("0")

def parse_cr(value):
    try:
        if "/" in value:
            num, denom = value.split("/")
            return float(num) / float(denom)
        return float(value)
    except:
        return None

df["challenge_rating_clean"] = df["challenge_rating_clean"].apply(parse_cr)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df["challenge_rating_clean"].dropna(), bins=20, kde=False, color="green", ax=ax)
ax.set_title("Distribution of Challenge Ratings")
ax.set_xlabel("Challenge Rating")
ax.set_ylabel("Count")
save_plot(fig, "challenge_rating_distribution.png")

print(f"✅ All plots saved in: {figures_path}")
