from .state import GameState
from .policies.rule_based import decide_response


def play_loop() -> None:
    state = GameState()
    print("Welcome to the AI Dungeon Master (Rule-Based MVP). Type 'quit' to exit.\n")
    print(f"Quest: {state.world.quest}")
    print(f"You are in the {state.world.location}.")

    end = False
    while not end:
        try:
            action = input("\nYour action> ")
        except EOFError:
            print("\nExiting.")
            break
        dm_text, end = decide_response(state, action)
        print(f"\nDM> {dm_text}")


