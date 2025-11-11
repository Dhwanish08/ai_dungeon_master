from typing import Tuple
from .state import GameState
from .policies.rule_based import decide_response
from .align_predictor import predict_alignment
from src.ai.gemini_client import generate_dm_reply


HELP_TEXT = "You hesitate. Would you like to (A) explore, (B) fight, (C) talk, or (D) inspect surroundings?"


EXPLORATION_KEYWORDS = {
    "go", "move", "walk", "run", "leave", "north", "south", "east", "west", "forest", "village", "ruins", "stairs"
}
FIGHT_KEYWORDS = {
    "attack", "fight", "strike", "swing", "hit", "use sword", "draw sword", "pull out sword", "bandit", "boss"
}
TALK_KEYWORDS = {"talk", "speak", "ask", "villager", "npc", "merchant", "shopkeeper"}
INSPECT_KEYWORDS = {"inspect", "examine", "look", "search", "track", "scout"}


def infer_intent(user_input: str) -> str | None:
    text = user_input.strip().lower()
    # Single-letter menu choices
    if text in {"a", "explore"}:
        return "explore"
    if text in {"b", "fight"}:
        return "fight"
    if text in {"c", "talk"}:
        return "talk"
    if text in {"d", "inspect", "look"}:
        return "inspect"
    if any(k in text for k in FIGHT_KEYWORDS):
        return "fight"
    if any(k in text for k in EXPLORATION_KEYWORDS):
        return "explore"
    if any(k in text for k in TALK_KEYWORDS):
        return "talk"
    if any(k in text for k in INSPECT_KEYWORDS):
        return "inspect"
    return None


GENERIC_LINES = {
    "the forest is quiet. birds scatter as you pass.",
    "time seems to stand still. try another action.",
}


def _summarize_state(state: GameState) -> str:
    parts = [
        f"location={state.world.location}",
        f"quest={state.world.quest}",
        f"flags={list(state.world.flags.keys())}",
        f"boss_active={state.world.boss_active}",
        f"players={[p.name + ':hp' + str(p.hp) for p in state.players]}",
    ]
    return "; ".join(parts)


def dm_step(state: GameState, user_input: str) -> Tuple[str, bool]:
    intent = infer_intent(user_input)
    dm_text, end_game = decide_response(state, user_input)

    # If policy gave a generic response and we couldn't infer clear intent, provide helpful options.
    if (intent is None) and (dm_text.strip().lower() in GENERIC_LINES or dm_text.startswith("You hesitate.")):
        return (HELP_TEXT, False)

    # Optional: call Gemini to narrate using our model guidance
    try:
        monster_alignment = None
        if state.world.boss_active:
            # Predict alignment for encounter flavor
            seed_name = state.world.story_seed.split(" ")[0] or "Forest Guardian"
            monster_alignment = predict_alignment(seed_name, "Large", 45.0, 14.0, 3.0)
        ai_text = generate_dm_reply(_summarize_state(state), user_input, intent or "unknown", monster_alignment)
        if ai_text:
            return (ai_text, end_game)
    except Exception:
        pass

    return dm_text, end_game


