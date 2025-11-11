import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib

# Lightweight, resilient glue that can operate without a trained classifier.
#
# Expected final API:
#  - load_model() -> model with:
#       predict_intent(text, state) -> (label:str, confidence:float)
#       predict_monster_behaviour(state) -> str | dict
#
# If not available, we provide a stub based on keywords and a debug JSON file.


DEBUG_JSON_PATH = Path("models/debug_intent_output.json")
INTENT_MODEL_PATH = Path("reports/artifacts/intent_model.joblib")
MONSTER_BEHAVIOR_MODEL_PATH = Path("reports/artifacts/monster_behavior_model.joblib")

FIGHT_WORDS = {"attack", "fight", "strike", "swing", "hit", "slash", "shoot", "cast at"}
EXPLORE_WORDS = {"explore", "go", "walk", "move", "scout", "search", "inspect", "look"}
TALK_WORDS = {"talk", "speak", "ask", "negotiate", "persuade"}
FLEE_WORDS = {"run", "flee", "escape", "retreat"}
INVENTORY_WORDS = {"inventory", "pack", "bag", "items"}


class _StubModel:
	def predict_intent(self, text: str, state: Dict[str, Any]) -> Tuple[str, float]:
		t = (text or "").strip().lower()
		if any(w in t for w in FIGHT_WORDS):
			return "attack", 0.85
		if any(w in t for w in TALK_WORDS):
			return "talk", 0.8
		if any(w in t for w in FLEE_WORDS):
			return "flee", 0.8
		if any(w in t for w in INVENTORY_WORDS):
			return "inventory", 0.75
		if any(w in t for w in EXPLORE_WORDS):
			return "explore", 0.7
		# try debug json
		if DEBUG_JSON_PATH.exists():
			try:
				data = json.loads(DEBUG_JSON_PATH.read_text())
				label = data.get("intent_label", "unknown")
				conf = float(data.get("intent_confidence", 0.5))
				return label, conf
			except Exception:
				pass
		return "unknown", 0.5

	def predict_monster_behaviour(self, state: Dict[str, Any]) -> Any:
		world = state.get("world", {}) or {}
		if not world.get("boss_active"):
			return {"action": "idle", "detail": "No active encounter."}
		
		# Try to load trained monster behavior model
		if MONSTER_BEHAVIOR_MODEL_PATH.exists():
			try:
				data = joblib.load(MONSTER_BEHAVIOR_MODEL_PATH)
				model = data["model"]
				le_location = data["le_location"]
				le_intent = data["le_intent"]
				le_behavior = data["le_behavior"]
				
				# Build feature vector: boss_hp, player_avg_hp, turn, location_encoded, intent_encoded
				boss_hp = int(world.get("boss_hp", 10))
				players = state.get("players", [])
				avg_hp = sum(p.get("hp", 10) for p in players) / len(players) if players else 10
				turn = int(world.get("turn", 1))
				location = world.get("location", "forest")
				# Get last intent from state if available, else default
				last_intent = state.get("last_intent", "attack")
				
				# Encode
				try:
					loc_enc = le_location.transform([location])[0]
				except:
					loc_enc = 0
				try:
					intent_enc = le_intent.transform([last_intent])[0]
				except:
					intent_enc = 0
				
				X = [[boss_hp, avg_hp, turn, loc_enc, intent_enc]]
				behavior_enc = model.predict(X)[0]
				behavior_label = le_behavior.inverse_transform([behavior_enc])[0]
				
				details = {
					"attack": "The foe lunges forward with deadly intent.",
					"defend": "The foe raises its guard, bracing for impact.",
					"taunt": "The foe snarls, trying to provoke you.",
					"retreat": "The foe backs away, looking for an escape.",
					"watch": "The foe watches warily, waiting for your move.",
				}
				return {"action": behavior_label, "detail": details.get(behavior_label, "The foe acts.")}
			except Exception as e:
				pass
		
		# Fallback to simple heuristics
		hp = int(world.get("boss_hp", 10))
		if hp <= 3:
			return {"action": "taunt", "detail": "The guardian staggers but refuses to yield."}
		return {"action": "attack", "detail": "The guardian presses the attack."}


class _SklearnIntentModel(_StubModel):
	def __init__(self, pipeline):
		self.pipeline = pipeline

	def predict_intent(self, text: str, state: Dict[str, Any]) -> Tuple[str, float]:
		if not text:
			return super().predict_intent(text, state)
		try:
			if hasattr(self.pipeline, "predict_proba"):
				proba = self.pipeline.predict_proba([text])[0]
				idx = int(proba.argmax())
				label = str(self.pipeline.classes_[idx])
				return label, float(proba[idx])
			# fallback to predict only
			label = str(self.pipeline.predict([text])[0])
			return label, 1.0
		except Exception:
			return super().predict_intent(text, state)


_MODEL: Optional[_StubModel] = None


def load_model():
	# Prefer trained intent model; fall back to heuristics.
	global _MODEL
	if _MODEL is None:
		if INTENT_MODEL_PATH.exists():
			try:
				pipeline = joblib.load(INTENT_MODEL_PATH)
				_MODEL = _SklearnIntentModel(pipeline)
			except Exception:
				_MODEL = _StubModel()
		else:
			_MODEL = _StubModel()
	return _MODEL


def get_intent_and_monster(text: str, game_state: Dict[str, Any]) -> Tuple[str, float, Any]:
	model = load_model()
	label, conf = model.predict_intent(text, game_state)
	# Store last intent in state for monster behavior prediction
	game_state["last_intent"] = label
	monster = model.predict_monster_behaviour(game_state)
	return label, conf, monster


