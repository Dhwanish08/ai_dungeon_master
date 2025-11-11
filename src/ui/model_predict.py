from typing import Any, Dict
import os
import joblib

from src.ui.intent_bridge import get_intent_and_monster


def predict(text: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Return a dict suitable for UI cards:
	{
	  "Player Intent": <label or placeholder>,
	  "Monster Behavior": <behavior or placeholder>
	}
	"""
	try:
		intent_label, intent_conf, monster_action = get_intent_and_monster(text, game_state)
		if isinstance(monster_action, dict):
			action = monster_action.get('action', 'idle')
			detail = monster_action.get('detail')
			monster_str = action if action else 'idle'
			if detail:
				monster_str += f" - {detail}"
		else:
			monster_str = monster_action if isinstance(monster_action, str) else str(monster_action)
		# If hostility model exists, refine monster behavior label
		hostility_path = os.path.join("reports", "artifacts", "hostility_model.joblib")
		if os.path.exists(hostility_path):
			try:
				model = joblib.load(hostility_path)  # this is a full pipeline
				# Build a minimal feature row from current game_state for prediction
				world = game_state.get("world", {}) or {}
				name = (world.get("story_seed") or "creature").split(" ")[0]
				size = "Medium"
				hp = world.get("boss_hp") or 10
				ac = 12
				cr = 1.0
				X = [{"name": name, "size": size, "hit_points": hp, "armor_class": ac, "challenge_rating_num": cr}]
				host_label = model.predict(X)[0]
				if str(host_label) == "hostile":
					monster_str = "Aggressive (likely to attack)"
				else:
					monster_str = "Passive (watching, non-hostile)"
			except Exception:
				pass
		return {
			"Player Intent": f"{intent_label or 'Unknown'} ({intent_conf:.2f})",
			"Monster Behavior": monster_str or "Idle",
		}
	except Exception:
		# Placeholder if live predictions are not available
		return {
			"Player Intent": "Exploration",
			"Monster Behavior": "Passive (waiting in shadows)",
		}


