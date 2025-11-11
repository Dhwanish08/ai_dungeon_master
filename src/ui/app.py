import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src.*` works when run via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.game.state import GameState
from src.game.loop import dm_step


st.set_page_config(page_title="AI Dungeon Master", page_icon="🧙", layout="wide")
st.title("🧙 D&D AI Dungeon Master")
st.caption("Local, no-API demo")


if "game_state" not in st.session_state:
    st.session_state.game_state = GameState()
    st.session_state.ended = False
    st.session_state.started = False
    st.session_state.num_players = 1

state: GameState = st.session_state.game_state

# Top controls: start game
st.markdown("---")
col_start1, col_start2 = st.columns([3, 1])
with col_start1:
    st.session_state.num_players = st.number_input("Enter # players", min_value=1, max_value=4, value=st.session_state.num_players, step=1)
    story_seed_input = st.text_input("Story seed (optional)", value=state.world.story_seed, placeholder="A cursed forest, a vanished baron, whispers under the ruins...")
with col_start2:
    if st.button("Start Game", type="primary"):
        state.world.story_seed = story_seed_input
        state.reset(num_players=st.session_state.num_players)
        state.add_log("**DM:** You wake at the tavern... The air smells of bread and ale.")
        st.session_state.started = True
        st.session_state.ended = False
        st.rerun()

st.markdown("---")

left, right = st.columns([2, 1])

with left:
    st.subheader("Story Window")
    chat = st.container()
    with chat:
        if not state.log:
            st.info("Press Start Game to begin.")
        for entry in state.log:
            st.markdown(entry)

    col1, col2 = st.columns([5, 1])
    with col1:
        user_text = st.text_input("Player Input", key="user_action", placeholder="explore tavern / talk to villager / go to forest ...", disabled=(not st.session_state.started or st.session_state.ended))
    with col2:
        send = st.button("Submit Action", use_container_width=True, disabled=(not st.session_state.started or st.session_state.ended))

def add_dm(text: str) -> None:
    state.add_log(f"**DM:** {text}")

def add_user(text: str) -> None:
    state.add_log(f"**You:** {text}")

if send and user_text.strip():
    add_user(user_text)
    dm_text, end_game = dm_step(state, user_text)
    add_dm(dm_text)
    st.session_state.ended = end_game
    st.rerun()

with right:
    st.subheader("Sidebar")
    st.markdown("**📍 Location**")
    st.write(state.world.location)
    st.markdown("**🧍‍♂️ Party status**")
    for p in state.players:
        st.write(f"{p.name}: HP {p.hp}")
    st.markdown("**🎒 Inventory**")
    inv = []
    for p in state.players:
        inv.extend(p.inventory)
    st.write(inv if inv else "(empty)")
    st.markdown("**🎲 Dice Log**")
    if state.dice_log:
        st.write("\n".join(state.dice_log[-10:]))
    else:
        st.write("(no rolls yet)")

    # Show encounter alignment if boss active
    if state.world.boss_active:
        try:
            from src.game.align_predictor import predict_alignment
            seed_name = state.world.story_seed.split(" ")[0] or "Forest Guardian"
            align = predict_alignment(seed_name, "Large", 45.0, 14.0, 3.0)
            if align:
                st.markdown("**🧭 Encounter Alignment (predicted)**")
                st.write(align)
        except Exception:
            pass

    st.markdown("**💾 Transcript**")
    if st.button("Download transcript"):
        transcript = "\n".join(state.log)
        st.download_button(
            label="Save transcript.txt",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )

st.divider()
colA, colB = st.columns(2)
with colA:
    if st.button("🔁 Restart Game", type="primary"):
        st.session_state.game_state = GameState()
        st.session_state.ended = False
        st.session_state.started = False
        st.rerun()
with colB:
    st.write(f"Turn: {state.world.turn}")
    st.write(f"Flags: {state.world.flags}")


