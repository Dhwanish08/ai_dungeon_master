from typing import Tuple
from ..state import GameState
from ..narrative import craft_narration
from ..align_predictor import predict_alignment


def decide_response(state: GameState, player_input: str, predicted_intent: str = None, intent_confidence: float = 0.0, monster_behavior: dict = None) -> Tuple[str, bool]:
    """Return (dm_text, end_game). Very simple rule-based logic with dice.
    Uses predicted_intent when text is ambiguous (intent_confidence >= 0.7).
    """
    text = player_input.strip().lower()
    state.world.turn += 1
    
    # Use ML intent prediction if confidence is high and text is ambiguous
    use_intent = predicted_intent and intent_confidence >= 0.7

    if text in {"quit", "exit"}:
        return ("The adventure ends for now. Farewell!", True)

    # Location-specific simple rules
    if state.world.location == "village":
        # Use intent prediction if high confidence
        if use_intent and predicted_intent == "talk":
            state.world.flags["rumor_bandits"] = True
            return (craft_narration("the village", "A villager whispers about bandits in the forest."), False)
        if use_intent and predicted_intent == "explore":
            state.world.location = "forest"
            return (craft_narration("the path north", "You head into the forest. The trees hush the wind."), False)
        
        # Original keyword matching
        if "talk" in text or "npc" in text or "villager" in text:
            state.world.flags["rumor_bandits"] = True
            return (craft_narration("the village", "A villager whispers about bandits in the forest."), False)
        if "shop" in text or "buy" in text:
            state.players[0].inventory.append("torch")
            return (craft_narration("the village", "The shopkeeper sells torches and rope. You purchase a torch."), False)
        if "forest" in text or "leave" in text or "north" in text:
            state.world.location = "forest"
            return (craft_narration("the path north", "You head into the forest. The trees hush the wind."), False)
        if any(k in text for k in ["inspect", "investigate", "explore", "look around"]):
            hint = "You could talk to villagers, buy supplies, or head north to the forest."
            return (craft_narration("the village square", hint), False)
        return (craft_narration("the village square", "You wander the village. People glance your way, expectant."), False)

    if state.world.location == "forest":
        if state.world.boss_active:
            # Combat loop - use intent prediction if high confidence
            should_attack = (use_intent and predicted_intent == "attack") or any(k in text for k in ["attack", "strike", "swing", "hit", "fight", "use sword"])
            should_flee = (use_intent and predicted_intent == "flee") or any(k in text for k in ["leave", "run", "escape", "south"])
            
            if should_attack:
                roll = state.roll()
                damage = 0
                if roll >= 15:
                    damage = 6
                elif roll >= 10:
                    damage = 3
                else:
                    # Monster behavior affects counterattack
                    monster_action = monster_behavior.get("action", "attack") if monster_behavior else "attack"
                    if monster_action == "defend":
                        return (craft_narration("the thicket", "Your attack glances off the foe's raised guard. It braces for your next strike.", roll=roll), False)
                    elif monster_action == "retreat":
                        return (craft_narration("the thicket", "Your attack misses as the foe backs away, looking for escape.", roll=roll), False)
                    return (craft_narration("the thicket", "Your attack misses as the foe lunges.", roll=roll), False)
                state.world.boss_hp = max(0, state.world.boss_hp - damage)
                if state.world.boss_hp == 0:
                    state.world.boss_active = False
                    state.world.flags["boss_defeated"] = True
                    state.players[0].inventory.append("silver coin")
                    return (craft_narration("the clearing", "The foe falls. You seize a silver coin. The way is open.", roll=roll), False)
                # Monster behavior affects response to damage
                monster_action = monster_behavior.get("action", "attack") if monster_behavior else "attack"
                if monster_action == "taunt":
                    return (craft_narration("the clearing", f"You wound the foe ({damage} dmg). It snarls defiantly: 'Is that all you have?'", roll=roll), False)
                elif monster_action == "retreat":
                    return (craft_narration("the clearing", f"You wound the foe ({damage} dmg). It staggers back, eyes darting for escape.", roll=roll), False)
                return (craft_narration("the clearing", f"You wound the foe ({damage} dmg). It snarls, blocking your way.", roll=roll), False)
            if should_flee:
                return (craft_narration("the clearing", "You cannot leave—the foe bars your path!"), False)
            return (craft_narration("the clearing", "The foe circles you. Attack or try something else."), False)

        if "search" in text or "track" in text or "footsteps" in text or "investigate" in text:
            roll = state.roll()
            if roll >= 12:
                state.world.flags["found_tracks"] = True
                return (craft_narration("the forest floor", "You find fresh tracks leading east toward ruins.", roll=roll), False)
            return (craft_narration("the underbrush", "You search but find only broken twigs.", roll=roll), False)
        if "east" in text or "ruins" in text:
            state.world.location = "ruins"
            return (craft_narration("the forest edge", "You arrive at mossy ruins. A dark stairway descends."), False)
        if "fight" in text or "bandit" in text:
            roll = state.roll()
            if roll >= 10:
                return (craft_narration("the thicket", "You fend off a lurking bandit and find a silver coin.", roll=roll), False)
            return (craft_narration("the thicket", "A bandit ambushes you. You lose 2 HP.", roll=roll), False)
        if any(k in text for k in ["leave", "south", "back", "village"]):
            # Trigger a blocking encounter unless boss already defeated
            if not state.world.flags.get("boss_defeated") and not state.world.boss_active:
                state.world.boss_active = True
                state.world.boss_hp = 10
                # Use alignment predictor to flavor encounter
                seed_name = state.world.story_seed.split(" ")[0] or "Forest Guardian"
                align = predict_alignment(seed_name, "Large", 45.0, 14.0, 3.0)
                if align == "good":
                    return (craft_narration("the clearing", "A guardian steps forth, bidding caution: 'Prove your intent before you pass.'"), False)
                if align == "neutral":
                    return (craft_narration("the clearing", "A horned sentinel watches silently, testing your resolve."), False)
                # default (evil or unknown)
                return (craft_narration("the clearing", "A horned shadow steps from the trees, blocking your path."), False)
            if state.world.flags.get("boss_defeated"):
                state.world.location = "village"
                return (craft_narration("the road", "You return to the village safely."), False)
        if "pull out sword" in text or "draw sword" in text:
            if "sword" not in state.players[0].inventory:
                state.players[0].inventory.append("sword")
            return (craft_narration("the forest", "You draw your sword, steel catching a pale light."), False)
        # Generic explore/inspect hints if nothing matched
        if any(k in text for k in ["inspect", "explore", "look around", "scout"]):
            if state.world.flags.get("found_tracks"):
                return (craft_narration("the forest path", "Tracks lead east toward the ruins."), False)
            if state.world.boss_active:
                return (craft_narration("the clearing", "The foe blocks your way—strike or create an advantage."), False)
            # chance to discover tracks on explore
            roll = state.roll()
            if roll >= 12:
                state.world.flags["found_tracks"] = True
                return (craft_narration("the forest floor", "You discover faint tracks heading east."), False)
            return (craft_narration("the trees", "You scout around but find little of note."), False)
        return ("The forest is quiet. Birds scatter as you pass.", False)

    if state.world.location == "ruins":
        if "down" in text or "descend" in text or "stair" in text:
            roll = state.roll()
            if roll >= 14:
                state.world.flags["amulet_found"] = True
                return (craft_narration("the stone stair", "You discover a hidden niche holding the lost amulet!", roll=roll), False)
            return (craft_narration("the stairwell", "You descend into darkness. The air smells of dust.", roll=roll), False)
        if "leave" in text or "back" in text:
            state.world.location = "forest"
            return (craft_narration("the ruined arch", "You climb back to the forest edge."), False)
        if "amulet" in text and state.world.flags.get("amulet_found"):
            return (craft_narration("your satchel", "You have the amulet. Return to the village to complete the quest."), False)
        if "village" in text:
            state.world.location = "village"
            if state.world.flags.get("amulet_found"):
                return (craft_narration("the village square", "You return the amulet. The village cheers. Quest complete!"), True)
            return (craft_narration("the village road", "You return to the village, empty-handed but determined."), False)
        return (craft_narration("the ruins", "Broken pillars surround a sunken hall. Whispers echo below."), False)

    return ("Time seems to stand still. Try another action.", False)


