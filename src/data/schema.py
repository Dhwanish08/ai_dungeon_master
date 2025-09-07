"""
Schema definitions for D&D datasets (Kaggle raw/processed version).
Directly matches Kaggle column names for simplicity.
"""

# Monster dataset schema
MONSTER_SCHEMA = {
    "Name": str,
    "Size": str,
    "Race + alignment": str,
    "HP": str,   # sometimes "12 (2d8+3)" so keep as string
    "Armor": str,
    "Speed": str,
    "Challenge rating  (XP)": str,  # text like "1 (200 XP)"
}

# Spell dataset schema
SPELL_SCHEMA = {
    "name": str,
    "level": int,
    "school": str,
    "cast_time": str,  # Kaggle used `cast_time`
    "range": str,
    "duration": str,
    "classes": str,    # classes that can cast this
    "verbal": str,
    "somatic": str,
    "material": str,
    "material_cost": str,
    "description\t\t\t\t": str,  # keep exactly as in file
}
