import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root on path for `import src.*` when run via `streamlit run src/ui/streamlit_app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.game.state import GameState  # type: ignore
from src.game.policies.rule_based import decide_response  # type: ignore
from src.ui.intent_bridge import get_intent_and_monster
from src.ui.gemini_client import generate_narration
from src.ui.model_predict import predict as predict_ui_dict


def _state_to_dict(state: GameState) -> Dict[str, Any]:
	return {
		"players": [
			{"name": p.name, "hp": p.hp, "inventory": list(p.inventory)}
			for p in state.players
		],
		"world": {
			"location": state.world.location,
			"turn": state.world.turn,
			"boss_active": state.world.boss_active,
			"boss_hp": state.world.boss_hp,
			"flags": dict(state.world.flags),
			"story_seed": state.world.story_seed,
			"quest": state.world.quest,
		},
		"dice_log": list(state.dice_log[-10:]),
	}


class GameSession:
	def __init__(self, state: GameState):
		self.state = state
		if not hasattr(self.state, "log"):
			self.state.log = []

	@property
	def story_log(self) -> List[str]:
		return getattr(self.state, "log", [])

	def append_log(self, who: str, text: str) -> None:
		self.state.add_log(f"{who}: {text}")

	def handle_group_action(self, player_indices: list[int], text: str) -> Tuple[str, Dict[str, Any]]:
		# Build a readable group name
		names = []
		for i in player_indices:
			if 0 <= i < len(self.state.players):
				names.append(self.state.players[i].name)
		group_name = " & ".join(names) if names else "All Players"

		game_state_dict = _state_to_dict(self.state)
		intent_label, intent_conf, monster_action = get_intent_and_monster(text, game_state_dict)

		# Pass intent prediction and monster behavior to engine so it can use ML guidance
		monster_dict = monster_action if isinstance(monster_action, dict) else {"action": str(monster_action)}
		engine_text, end_game = decide_response(self.state, text, predicted_intent=intent_label, intent_confidence=intent_conf, monster_behavior=monster_dict)

		action_summary = f"{group_name}: {engine_text}"
		narration = generate_narration(
			game_state=_state_to_dict(self.state),
			recent_player_action=f"[{group_name}] {text}",
			action_summary=action_summary,
		)

		# Single combined log line for actors
		self.append_log(group_name, text)
		self.append_log("DM", narration)

		ui_predictions = predict_ui_dict(text, _state_to_dict(self.state))

		intent_applied = intent_conf >= 0.7
		monster_used = bool(self.state.world.boss_active and monster_dict.get("action") not in {"idle", "watch"})
		effects = []
		if intent_label:
			if intent_applied:
				effects.append(f"Intent '{intent_label}' routed this action.")
			else:
				effects.append(f"Intent '{intent_label}' noted (confidence {intent_conf:.2f}).")
		if monster_used:
			action = monster_dict.get("action", "attack")
			detail = monster_dict.get("detail")
			msg = f"Monster prepares to {action}."
			if detail:
				msg += f" {detail}"
			effects.append(msg)

		panel = {
			"intent_label": intent_label,
			"intent_confidence": intent_conf,
			"intent_applied": intent_applied,
			"monster_action": monster_dict,
			"monster_used": monster_used,
			"monster_detail": monster_dict.get("detail"),
			"engine_outcome": engine_text,
			"ended": end_game,
			"predictions": ui_predictions,
			"actors": names,
			"effect_message": " | ".join(effects) if effects else None,
		}
		return narration, panel

	def handle_player_action(self, player_idx: int, text: str) -> Tuple[str, Dict[str, Any]]:
		game_state_dict = _state_to_dict(self.state)

		# Local ML: intent + monster behaviour
		intent_label, intent_conf, monster_action = get_intent_and_monster(text, game_state_dict)

		# Deterministic engine outcome - now uses ML intent and monster behavior when confidence is high
		monster_dict = monster_action if isinstance(monster_action, dict) else {"action": str(monster_action)}
		engine_text, end_game = decide_response(self.state, text, predicted_intent=intent_label, intent_confidence=intent_conf, monster_behavior=monster_dict)

		# Narration strictly based on engine outcome (exact Gemini prompt is set inside the client)
		player_name = self.state.players[player_idx].name if 0 <= player_idx < len(self.state.players) else f"Player {player_idx+1}"
		action_summary = f"{player_name}: {engine_text}"
		narration = generate_narration(
			game_state=_state_to_dict(self.state),
			recent_player_action=f"[{player_name}] {text}",
			action_summary=action_summary,
		)

		# Log player with actual name and narrated DM text (model outputs only in side panel)
		self.append_log(player_name, text)
		self.append_log("DM", narration)

		# UI predictions dict for right-side cards
		ui_predictions = predict_ui_dict(text, _state_to_dict(self.state))

		intent_applied = intent_conf >= 0.7
		monster_used = bool(self.state.world.boss_active and monster_dict.get("action") not in {"idle", "watch"})
		effects = []
		if intent_label:
			if intent_applied:
				effects.append(f"Intent '{intent_label}' routed this action.")
			else:
				effects.append(f"Intent '{intent_label}' noted (confidence {intent_conf:.2f}).")
		if monster_used:
			action = monster_dict.get("action", "attack")
			detail = monster_dict.get("detail")
			msg = f"Monster prepares to {action}."
			if detail:
				msg += f" {detail}"
			effects.append(msg)

		panel = {
			"intent_label": intent_label,
			"intent_confidence": intent_conf,
			"intent_applied": intent_applied,
			"monster_action": monster_dict,
			"monster_used": monster_used,
			"monster_detail": monster_dict.get("detail"),
			"engine_outcome": engine_text,
			"ended": end_game,
			"predictions": ui_predictions,
			"effect_message": " | ".join(effects) if effects else None,
		}
		return narration, panel

