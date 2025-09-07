import pandas as pd
import os

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def load_monsters():
    path = os.path.join(RAW_DIR, "dd5e_monsters.csv")
    df = pd.read_csv(path)
    return df

def load_spells():
    path = os.path.join(RAW_DIR, "dnd-spells.csv")
    df = pd.read_csv(path)
    return df

def save_processed(df, filename):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved processed file: {path}")

if __name__ == "__main__":
    monsters = load_monsters()
    spells = load_spells()

    print("📊 Monsters shape:", monsters.shape)
    print("📊 Spells shape:", spells.shape)

    # Save copies in processed/
    save_processed(monsters, "monsters_clean.csv")
    save_processed(spells, "spells_clean.csv")
