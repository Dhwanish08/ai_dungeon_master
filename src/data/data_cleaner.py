import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_monsters():
    df = pd.read_csv(RAW_DIR / "Dd5e_monsters.csv")

    # Rename to match schema
    rename_map = {
        "Name": "name",
        "Size": "size",
        "Race + alignment": "alignment",
        "Armor": "armor_class",
        "HP": "hit_points",
    }
    df = df.rename(columns=rename_map)

    # Add missing fields with defaults
    for col in ["type", "hit_dice", "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]:
        if col not in df.columns:
            df[col] = None

    # Save
    df.to_csv(PROCESSED_DIR / "monsters.csv", index=False)
    print("✅ Cleaned monsters saved.")

def clean_spells():
    df = pd.read_csv(RAW_DIR / "dnd-spells.csv")

    rename_map = {
        "name": "name",
        "level": "level",
        "school": "school",
        "cast_time": "casting_time",
        "range": "range",
        "duration": "duration",
    }
    df = df.rename(columns=rename_map)

    # Combine components
    df["components"] = df[["verbal", "somatic", "material"]].astype(str).agg(", ".join, axis=1)

    # Clean description (remove tabs/spaces)
    if "description\t\t\t\t" in df.columns:
        df["description"] = df["description\t\t\t\t"].astype(str).str.strip()
        df = df.drop(columns=["description\t\t\t\t"])

    # Drop old component cols
    for col in ["verbal", "somatic", "material", "material_cost", "classes"]:
        if col in df.columns:
            df = df.drop(columns=col)

    df.to_csv(PROCESSED_DIR / "spells.csv", index=False)
    print("✅ Cleaned spells saved.")

if __name__ == "__main__":
    clean_monsters()
    clean_spells()
