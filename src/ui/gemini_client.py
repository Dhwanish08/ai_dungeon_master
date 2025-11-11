import os
from typing import Any, Dict, Optional

from . import gemini_fallback


SYSTEM_INSTRUCTIONS = (
	"System:\n"
	"You are the Dungeon Master narrator. You will describe events, atmospheres, NPC dialogue, and hints. "
	"You must not decide combat outcomes, damage, or game-rule changes — those are determined by the game engine. "
	"The engine will supply the action outcome summary and current game state. Your job is to craft immersive narration based on that outcome. "
	"If player input is confusing or nonsense, convert it into a meaningful narrative hook (e.g., “Your clumsy swing startles a fox; in the distance, a torch flares — perhaps someone is near”) "
	"and give 2–3 clear suggestions (Fight / Explore / Talk / Inspect). Always include a short question to invite the next action. "
	"Keep responses 2–4 sentences. Use tone: Mystical, dramatic storyteller.\n\n"
	"You are a cinematic Dungeon Master narrating a dynamic fantasy RPG. "
	"Use second-person narration (\"You step into the forest...\"). "
	"If a player’s action is vague or unrelated, guide them gently back into the quest with clues, side characters, or narrative hints. "
	"Include all player names in relevant moments. "
	"Keep tone immersive, short, and story-like — never list actions or steps.\n\n"
	"User (provide programmatic fields):\n\n"
	"game_state: JSON containing location, players, HP, inventory, flags, turn.\n\n"
	"action_summary: short string describing the engine result: e.g. \"Player 1 attacked goblin; roll 19; goblin HP reduced to 0; loot silver coin.\"\n\n"
	"recent_player_text: raw text the player typed.\n\n"
	"Assistant should output: A small paragraph (2–4 sentences) narrating the scene, with a trailing suggestion like: "
	"“Will you strike again, search the clearing, or try to persuade them?”"
)


def _call_gemini_stub(prompt: str) -> Optional[str]:
	"""
	Lightweight placeholder so the app runs without extra deps.
	If GEMINI_API_KEY is present but no SDK is installed, we gracefully return None.
	"""
	# Intentionally not importing any SDK to keep requirements simple.
	# In a real setup, import Google's Gemini SDK and call here.
	return None

def generate_narration(
	game_state: Dict[str, Any],
	recent_player_action: str = "",
	action_summary: Optional[str] = None,
) -> str:
	"""
	Generate a short DM paragraph. If GEMINI_API_KEY is missing or any error occurs,
	falls back to deterministic narration.
	"""
	api_key = os.environ.get("GEMINI_API_KEY")
	try:
		if api_key:
			prompt = (
				f"{SYSTEM_INSTRUCTIONS}\n\n"
				f"game_state: {game_state}\n\n"
				f"action_summary: {action_summary or 'N/A'}\n\n"
				f"recent_player_text: {recent_player_action}\n"
			)
			resp = _call_gemini_stub(prompt)
			if resp and resp.strip():
				return resp.strip()
	except Exception:
		pass

	return gemini_fallback.generate_narration(
		game_state=game_state,
		recent_player_action=recent_player_action,
		action_summary=action_summary,
	)


