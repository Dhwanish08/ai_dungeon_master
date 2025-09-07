"""
Cleaned schema definitions for D&D datasets.
Maps weird Kaggle column names into cleaner, consistent ones.
"""

# Monster dataset (cleaned)
MONSTER_CLEAN_SCHEMA = {
    "name": str,
    "size": str,
    "type_alignment": str,  # merged Race + alignment
    "hit_points": str,
    "armor_class": str,
    "speed": str,
    "challenge_rating": str,
}

# Spell dataset (cleaned)
SPELL_CLEAN_SCHEMA = {
    "name": str,
    "level": int,
    "school": str,
    "casting_time": str,  # renamed from cast_time
    "range": str,
    "duration": str,
    "classes": str,
    "verbal": str,
    "somatic": str,
    "material": str,
    "material_cost": str,
    "description": str,
}
