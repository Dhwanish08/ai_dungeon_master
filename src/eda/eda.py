import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROCESSED_PATH = "data/processed"

def load_data():
    monsters = pd.read_csv(os.path.join(PROCESSED_PATH, "Dd5e_monsters_clean.csv"))
    spells = pd.read_csv(os.path.join(PROCESSED_PATH, "dnd_spells_clean.csv"))
    return monsters, spells

def eda_monsters(monsters):
    print("\n--- Monsters Dataset ---")
    print(monsters.info())
    print(monsters.describe())

    # Distribution of hit points
    plt.figure(figsize=(8,5))
    sns.histplot(monsters["hit_points"], bins=30, kde=True)
    plt.title("Distribution of Monster HP")
    plt.savefig("data/processed/monster_hp_dist.png")
    plt.close()

    # Average HP by Size
    plt.figure(figsize=(8,5))
    monsters.groupby("size")["hit_points"].mean().plot(kind="bar")
    plt.title("Average Monster HP by Size")
    plt.ylabel("Average HP")
    plt.savefig("data/processed/monster_hp_by_size.png")
    plt.close()

def eda_spells(spells):
    print("\n--- Spells Dataset ---")
    print(spells.info())
    print(spells.describe())

    # Spell count by level
    plt.figure(figsize=(8,5))
    spells["level"].value_counts().sort_index().plot(kind="bar")
    plt.title("Number of Spells per Level")
    plt.xlabel("Level")
    plt.ylabel("Count")
    plt.savefig("data/processed/spells_per_level.png")
    plt.close()

    # Most common spell schools
    plt.figure(figsize=(8,5))
    spells["school"].value_counts().head(10).plot(kind="bar")
    plt.title("Top 10 Spell Schools")
    plt.savefig("data/processed/spell_schools.png")
    plt.close()

if __name__ == "__main__":
    monsters, spells = load_data()
    eda_monsters(monsters)
    eda_spells(spells)
    print("✅ EDA complete. Results saved in data/processed/")
