import random
from typing import Optional


ADJECTIVES = [
    "shadowed",
    "wind-swept",
    "mossy",
    "ancient",
    "dust-choked",
    "candle-lit",
    "soggy",
    "lonely",
    "bustling",
]

SENSES = [
    "the air smells of {smell}",
    "you hear {sound} in the distance",
    "a {sight} catches your eye",
]

SMELLS = ["bread and ale", "wet stone", "old parchment", "pine and ash"]
SOUNDS = ["soft whispers", "distant footsteps", "dripping water", "rustling leaves"]
SIGHTS = ["glint of metal", "faint glow", "fluttering moth", "curl of smoke"]

VERBS = ["drifts", "lingers", "presses", "creeps", "flickers"]


def _sensory_line() -> str:
    tmpl = random.choice(SENSES)
    return tmpl.format(smell=random.choice(SMELLS), sound=random.choice(SOUNDS), sight=random.choice(SIGHTS))


def craft_narration(scene: str, base: str, roll: Optional[int] = None) -> str:
    parts = []
    descriptor = random.choice(ADJECTIVES)
    parts.append(f"{scene.capitalize()} feels {descriptor}.")
    parts.append(base)
    if random.random() < 0.7:
        parts.append(_sensory_line())
    if roll is not None:
        parts.append(f"(DM rolls d20: {roll})")
    return " " .join(parts)


