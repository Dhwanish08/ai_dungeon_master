import random
from typing import Any, Dict, Optional


def _variation() -> str:
	opens = [
		"The air carries a hush,",
		"A distant crow calls as",
		"Lantern-light flickers while",
		"A breeze ripples the leaves as",
		"Your boots crunch on old stone while",
	]
	twists = [
		"shadows stretch like wary sentries.",
		"the scent of moss and iron lingers.",
		"somewhere, a wooden sign creaks.",
		"footprints fade into the undergrowth.",
		"a bell tolls once, then falls silent.",
	]
	return f"{random.choice(opens)} {random.choice(twists)}"


def generate_narration(
	game_state: Dict[str, Any],
	recent_player_action: str,
	action_summary: Optional[str] = None,
) -> str:
	"""
	Deterministic-ish fallback narrator: 2–4 sentences describing the outcome provided.
	The narrator never decides outcomes; it only dresses the passed-in outcome and state.
	"""
	location = (game_state.get("world", {}) or {}).get("location", "unknown place")
	turn = (game_state.get("world", {}) or {}).get("turn", 0)
	party = game_state.get("players", [])
	party_status = ", ".join(f"{p.get('name','Adventurer')} (HP {p.get('hp','?')})" for p in party) or "your party"

	hook = _variation()
	outcome_line = action_summary or "Events unfold as determined by fate."
	suggestion = "Will you strike again, search the area, or try to talk?"

	lines = [
		f"Turn {turn} at the {location}. {hook}",
		f"You act: {recent_player_action.strip() or '...'}",
		f"Outcome: {outcome_line}",
		f"Party: {party_status}. {suggestion}",
	]
	# Keep it concise: 2–4 sentences
	return " ".join(lines[: random.choice([2, 3, 4])])


