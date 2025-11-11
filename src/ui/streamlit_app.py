import base64
import sys
from pathlib import Path

# Ensure project root on sys.path so `import src.*` works in Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.game.state import GameState
from src.ui.game_session import GameSession


def _load_base64(path: Path) -> str:
	try:
		return base64.b64encode(path.read_bytes()).decode("utf-8")
	except Exception:
		return ""


ASSETS_DIR = Path(__file__).resolve().parent
MAIN_BG = _load_base64(ASSETS_DIR / "wp2770223-dd-wallpaper.jpg")
STATS_BG = _load_base64(ASSETS_DIR / "wp2770226-dd-wallpaper.jpg")


st.set_page_config(page_title="AI Dungeon Master (Hybrid Prototype)", page_icon="🧙", layout="wide")

# Dark futuristic style with gentle fades + custom imagery
DARK_CSS = f"""
<style>
  .stApp {{
    background-image:
      linear-gradient(135deg, rgba(5,7,11,0.88), rgba(7,10,16,0.92)),
      url("data:image/jpeg;base64,{MAIN_BG}");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    color: #e0e6ef;
  }}
  .block-container {{ padding-top: 1.25rem; }}
  .panel {{
    background: rgba(12, 15, 22, 0.82);
    border: 1px solid #1f2530;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    box-shadow: 0 0 24px rgba(0,0,0,0.35);
    backdrop-filter: blur(2px);
  }}
  .story-bg {{
    background-image:
      linear-gradient(135deg, rgba(9,12,18,0.88), rgba(10,14,21,0.75)),
      url("data:image/jpeg;base64,{STATS_BG}");
    background-size: cover;
    background-position: center;
  }}
  .story-log {{ max-height: 60vh; overflow-y: auto; }}
  .fade-in {{ animation: fadeIn 350ms ease-out; }}
  @keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .card {{
    border: 1px solid #263043;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.5rem;
    background: rgba(16, 20, 30, 0.82);
    backdrop-filter: blur(2px);
  }}
  .stats-card {{
    background-image:
      linear-gradient(180deg, rgba(7,10,16,0.82), rgba(6,9,14,0.92)),
      url("data:image/jpeg;base64,{STATS_BG}");
    background-size: cover;
    background-position: center;
    color: #dfe6f7;
    box-shadow: 0 0 20px rgba(0,0,0,0.3);
  }}
  .accent {{ color: #6ad0ff; }}
  .hint {{ color: #b8c7e0; font-size: 0.9rem; }}
  .top-scene {{
    background: linear-gradient(135deg, rgba(16,20,30,0.88), rgba(8,12,20,0.88));
    border: 1px solid #1b2330;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 0 18px rgba(0,0,0,0.25);
  }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.45rem;
    border-radius: 999px;
    font-size: 0.75rem;
    margin-top: 0.35rem;
    margin-right: 0.35rem;
  }}
  .badge-active {{
    background: #43e18c;
    color: #05131d;
    font-weight: 600;
  }}
  .badge-muted {{
    background: rgba(90, 111, 146, 0.6);
    color: #d4ddf2;
  }}
  .model-banner {{
    background: rgba(67, 225, 140, 0.12);
    border: 1px solid rgba(67, 225, 140, 0.35);
    color: #7bf3a3;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.75rem;
    font-size: 0.92rem;
    box-shadow: 0 0 20px rgba(0,0,0,0.25);
  }}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

st.title("🧙 AI Dungeon Master — Hybrid Prototype")
st.caption("Local model: intent/monster logic; Gemini: narration only.")

if "game_state" not in st.session_state:
	st.session_state.game_state = GameState()
	st.session_state.started = False
	st.session_state.ended = False
	st.session_state.num_players = 1
	st.session_state.player_names = []
	st.session_state.last_panel = None
	st.session_state.active_player_idx = 0

state: GameState = st.session_state.game_state
session = GameSession(state)

with st.sidebar:
	st.header("Party & World")
	st.markdown(f"**📍 Location:** <span class='accent'>{state.world.location}</span>", unsafe_allow_html=True)
	st.markdown("**🧍 Party**")
	for i, p in enumerate(state.players):
		st.markdown(f"- {p.name}: **HP {p.hp}** | items: {', '.join(p.inventory) or '(none)'}")

	st.markdown("**🎲 Dice Log**")
	if state.dice_log:
		st.code("\n".join(state.dice_log[-8:]))
	else:
		st.write("(no rolls)")

	st.markdown("---")
	st.subheader("Local Model Outputs")
	panel = st.session_state.get('last_panel')
	if panel:
		intent_label = panel.get('intent_label', 'unknown')
		intent_conf = panel.get('intent_confidence', 0.0)
		intent_line = f"Intent: **{intent_label}** ({intent_conf:.2f})"
		status = "Applied to engine" if panel.get('intent_applied') else "Reference only"
		st.markdown(f"{intent_line} — {status}")
		if panel.get('monster_used'):
			monster = panel.get('monster_action', {}) or {}
			action = monster.get('action', 'unknown')
			detail = monster.get('detail')
			monster_line = f"Monster: **{action}**"
			if detail:
				monster_line += f" — {detail}"
			st.markdown(monster_line)
	else:
		st.write("(no predictions yet)")

with st.container():
	st.subheader("Start")
	col1, col2 = st.columns([3, 1])
	with col1:
		st.session_state.num_players = st.number_input("Number of players", 1, 4, st.session_state.num_players, step=1)
		seed = st.text_input("Story seed (optional)", value=state.world.story_seed)
		# Player names inputs
		names_cols = st.columns(st.session_state.num_players)
		new_names = []
		for i in range(st.session_state.num_players):
			with names_cols[i]:
				new_names.append(st.text_input(f"Player {i+1} name", key=f"pname_{i}", value=(st.session_state.player_names[i] if i < len(st.session_state.player_names) else f"Player {i+1}")))
		st.session_state.player_names = new_names
	with col2:
		if st.button("Start Game", type="primary"):
			state.reset(num_players=st.session_state.num_players)
			state.world.story_seed = seed
			# Set player names
			for i, name in enumerate(st.session_state.player_names[:len(state.players)]):
				state.players[i].name = name or f"Player {i+1}"
			st.session_state.started = True
			st.session_state.ended = False
			st.session_state.last_panel = None
			st.session_state.active_player_idx = 0
			party_names = ", ".join(p.name for p in state.players)
			session.append_log("DM", f"{party_names} gather as dusk falls over the old road. A cold wind hints at secrets beyond the village.")
			st.rerun()

st.markdown("---")
left, right = st.columns([2, 1])

with left:
	# Top scene/location
	st.markdown(f"<div class='top-scene'><strong>Scene</strong>: <span class='accent'>{state.world.location}</span> &nbsp; • &nbsp; Turn {state.world.turn}</div>", unsafe_allow_html=True)
	last_panel = st.session_state.get('last_panel')
	if last_panel and last_panel.get('effect_message'):
		st.markdown(f"<div class='model-banner'>{last_panel['effect_message']}</div>", unsafe_allow_html=True)
	st.subheader("Story Log")
	# Active player selector
	if st.session_state.started and not st.session_state.ended and state.players:
		player_options = list(range(len(state.players))) + [-1]
		def _format(idx: int) -> str:
			if idx == -1:
				return "All Players"
			return state.players[idx].name

		current_index = player_options.index(st.session_state.active_player_idx) if st.session_state.active_player_idx in player_options else 0
		selected = st.selectbox(
			"Current player",
			options=player_options,
			index=current_index,
			format_func=_format,
		)
		st.session_state.active_player_idx = selected
	log_box = st.container()
	with log_box:
		st.markdown('<div class="panel story-log story-bg">', unsafe_allow_html=True)
		if not session.story_log:
			st.info("Press Start Game to begin.")
		else:
			for entry in session.story_log:
				st.markdown(f"<div class='fade-in'>{entry}</div>", unsafe_allow_html=True)
		st.markdown("</div>", unsafe_allow_html=True)

	# Quick actions
	st.markdown("Quick Actions")
	qa1, qa2, qa3, qa4, qa5 = st.columns(5)
	def submit_action(text: str):
		if not st.session_state.started or st.session_state.ended:
			return
		idx = st.session_state.active_player_idx or 0
		if idx == -1:
			# All players act together: single engine step and combined narration
			_, panel = session.handle_group_action(list(range(len(state.players))), text)
			st.session_state.last_panel = panel
			st.session_state.ended = panel.get("ended", False)
			if not st.session_state.ended:
				st.session_state.active_player_idx = 0
			st.rerun()
			return
		idx = int(idx)
		_, panel = session.handle_player_action(idx, text)
		st.session_state.last_panel = panel
		st.session_state.ended = panel.get("ended", False)
		# advance turn to next player
		if not st.session_state.ended and state.players:
			st.session_state.active_player_idx = (idx + 1) % len(state.players)
		st.rerun()
	with qa1:
		st.button("⚔️ Attack", on_click=lambda: submit_action("attack the threat"))
	with qa2:
		st.button("🧭 Explore", on_click=lambda: submit_action("explore the area"))
	with qa3:
		st.button("🗣️ Talk", on_click=lambda: submit_action("talk to the nearest NPC"))
	with qa4:
		st.button("🎒 Inventory", on_click=lambda: submit_action("check inventory"))
	with qa5:
		st.button("🏃 Run", on_click=lambda: submit_action("flee back to safety"))

	# Free-form input
	col_in1, col_in2 = st.columns([5, 1])
	with col_in1:
		user_text = st.text_input(
			"Your action",
			key="user_action",
			placeholder="ex: inspect ruins / talk to villager / draw sword",
			disabled=(not st.session_state.started or st.session_state.ended),
		)
	with col_in2:
		submit = st.button("Submit", use_container_width=True, disabled=(not st.session_state.started or st.session_state.ended))

	if submit and user_text.strip():
		idx = st.session_state.active_player_idx or 0
		if idx == -1:
			_, panel = session.handle_group_action(list(range(len(state.players))), user_text)
			st.session_state.last_panel = panel
			st.session_state.ended = panel.get("ended", False)
			if not st.session_state.ended:
				st.session_state.active_player_idx = 0
			st.rerun()
		else:
			idx = int(idx)
			_, panel = session.handle_player_action(idx, user_text)
			st.session_state.last_panel = panel
			st.session_state.ended = panel.get("ended", False)
			# advance turn
			if not st.session_state.ended and state.players:
				st.session_state.active_player_idx = (idx + 1) % len(state.players)
			st.rerun()

with right:
	st.subheader("Model Panel")
	last_action_entry = next((e for e in reversed(session.story_log) if not e.startswith('DM:')), None)
	panel = st.session_state.get('last_panel') or {}
	intent_label = panel.get('intent_label', '(none)')
	intent_conf = panel.get('intent_confidence', 0.0)
	intent_line = f"{intent_label} ({intent_conf:.2f})" if panel else "(none)"
	intent_badge = "<span class='badge badge-active'>Applied to engine</span>" if panel.get('intent_applied') else "<span class='badge badge-muted'>Reference only</span>"
	st.markdown("<div class='card stats-card'><strong>Last Action</strong><br/><span class='hint'>{}</span></div>".format(last_action_entry or "(none)"), unsafe_allow_html=True)
	st.markdown("<div class='card stats-card'><strong>Player Intent</strong><br/><span class='hint'>{}</span><br/>{}</div>".format(intent_line, intent_badge), unsafe_allow_html=True)
	monster_card_shown = False
	if panel.get('monster_action'):
		monster = panel.get('monster_action') or {}
		action = monster.get('action', 'idle')
		detail = monster.get('detail')
		monster_line = f"{action}" if action else "(none)"
		if detail:
			monster_line += f" — {detail}"
		badge = "<span class='badge badge-active'>Used in encounter</span>" if panel.get('monster_used') else "<span class='badge badge-muted'>No active encounter</span>"
		st.markdown("<div class='card stats-card'><strong>Monster Behavior</strong><br/><span class='hint'>{}</span><br/>{}</div>".format(monster_line, badge), unsafe_allow_html=True)
		monster_card_shown = True
	if not monster_card_shown and panel:
		st.markdown("<div class='card stats-card'><strong>Monster Behavior</strong><br/><span class='hint'>(no active encounter)</span></div>", unsafe_allow_html=True)

	st.markdown("---")
	st.markdown("**Controls**")
	colA, colB = st.columns(2)
	with colA:
		if st.button("🔁 Restart"):
			st.session_state.game_state = GameState()
			st.session_state.started = False
			st.session_state.ended = False
			st.session_state.last_panel = None
			st.rerun()
	with colB:
		st.write(f"Turn: {state.world.turn}")
		st.write(f"Flags: {state.world.flags}")


