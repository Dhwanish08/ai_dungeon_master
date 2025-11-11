from dataclasses import dataclass, field
from typing import List, Dict
import random


@dataclass
class Player:
    name: str
    hp: int = 10
    inventory: List[str] = field(default_factory=list)


@dataclass
class WorldState:
    location: str = "village"
    quest: str = "Find the lost amulet"
    danger_level: int = 1
    turn: int = 0
    flags: Dict[str, bool] = field(default_factory=dict)
    story_seed: str = ""
    boss_active: bool = False
    boss_hp: int = 0


@dataclass
class GameState:
    players: List[Player] = field(default_factory=lambda: [Player(name="Hero")])
    world: WorldState = field(default_factory=WorldState)
    log: List[str] = field(default_factory=list)
    dice_log: List[str] = field(default_factory=list)

    def roll(self, sides: int = 20) -> int:
        value = random.randint(1, sides)
        self.dice_log.append(f"d{sides}: {value}")
        return value

    def add_log(self, entry: str) -> None:
        self.log.append(entry)

    def reset(self, num_players: int = 1) -> None:
        self.players = [Player(name=f"Player {i+1}") for i in range(max(1, num_players))]
        seed = self.world.story_seed  # carry over seed on restart
        self.world = WorldState(story_seed=seed)
        self.log = []
        self.dice_log = []


