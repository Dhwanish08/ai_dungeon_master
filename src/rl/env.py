import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict
from ..game.state import GameState
from ..game.policies.rule_based import decide_response


class DungeonMasterEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.state = GameState()
        # Discrete action space as placeholder over a small set of DM intents
        self.intents = [
            "narrate",
            "hint",
            "escalate",
            "reward",
        ]
        self.action_space = spaces.Discrete(len(self.intents))
        # Observation: a simple vector with location/danger/flags count
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.int32)
        self.last_player_action = "start"

    def _obs(self) -> np.ndarray:
        loc_code = {"village": 0, "forest": 1, "ruins": 2}.get(self.state.world.location, 0)
        return np.array([
            loc_code,
            int(self.state.world.danger_level),
            int(sum(self.state.world.flags.values())),
        ], dtype=np.int32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self.state = GameState()
        self.last_player_action = "start"
        return self._obs(), {}

    def step(self, action: int):
        # Map intent to a synthetic player action to drive the world via rule-based engine
        intent = self.intents[action]
        synthetic_player = {
            "narrate": "look around",
            "hint": "talk to villager",
            "escalate": "go to forest",
            "reward": "descend stairs",
        }.get(intent, "look around")

        _, done = decide_response(self.state, synthetic_player)

        reward = 0.0
        if self.state.world.flags.get("amulet_found"):
            reward += 1.0
        if done:
            reward += 1.0

        obs = self._obs()
        info = {}
        return obs, reward, done, False, info

    def render(self):
        return f"Loc={self.state.world.location}, Flags={self.state.world.flags}"


