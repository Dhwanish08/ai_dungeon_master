import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

# Monster cleaning
def clean_monsters():
    df = pd.read_csv(os.path.join(RAW_PATH, "Dd5e_monsters.csv"))

    # Rename columns to match schema
    df = df.rename(columns={
        "Name": "name",
        "Size": "size",
        "Race + alignment": "alignment",
        "Armor": "armor_class",
        "HP": "hit_points",
        "Speed": "speed",
        "Challenge rating  (XP)": "challenge_rating"
    })

    # Convert HP to int (where possible)
    df["hit_points"] = pd.to_numeric(df["hit_points"], errors="coerce")

    # Convert Armor to int
    df["armor_class"] = pd.to_numeric(df["armor_class"], errors="coerce")

    # Drop duplicates
    df = df.drop_duplicates()

    # Save cleaned version
    df.to_csv(os.path.join(PROCESSED_PATH, "Dd5e_monsters_clean.csv"), index=False)
    print("✅ Cleaned monster dataset saved.")


# Spell cleaning
def clean_spells():
    df = pd.read_csv(os.path.join(RAW_PATH, "dnd-spells.csv"))

    # Rename to match schema
    df = df.rename(columns={
        "name": "name",
        "level": "level",
        "school": "school",
        "cast_time": "casting_time",
        "range": "range",
        "verbal": "verbal",
        "somatic": "somatic",
        "material": "material",
        "material_cost": "material_cost",
        "duration": "duration",
        "description\t\t\t\t": "description"
    })

    # Fill missing values with "Unknown"
    df = df.fillna("Unknown")

    # Drop duplicates
    df = df.drop_duplicates()

    # Save cleaned version
    df.to_csv(os.path.join(PROCESSED_PATH, "dnd_spells_clean.csv"), index=False)
    print("✅ Cleaned spells dataset saved.")


if __name__ == "__main__":
    clean_monsters()
    clean_spells()
