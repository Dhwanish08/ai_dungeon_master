import os
from typing import Optional


_client = None
_model = None


def _init_client():
    global _client, _model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _client = genai
        _model = genai.GenerativeModel("gemini-1.5-flash")
        return _model
    except Exception:
        return None


def generate_dm_reply(state_summary: str, user_input: str, intent: str, monster_alignment: Optional[str]) -> Optional[str]:
    global _model
    if _model is None:
        _model = _init_client()
    if _model is None:
        return None

    system_prompt = (
        "You are an AI Dungeon Master. Keep responses concise (2-5 sentences), descriptive, and actionable. "
        "Respect world state and intent guidance. If combat is ongoing, keep turns tight. Avoid meta-talk."
    )
    guidance = (
        f"Intent: {intent}. "
        f"Monster alignment (if any): {monster_alignment or 'unknown'}. "
        "Tone: If alignment is evil -> aggressive; neutral -> cautious; good -> measured and fair."
    )
    prompt = (
        f"World State: {state_summary}\n"
        f"Player: {user_input}\n"
        f"Guidance: {guidance}\n"
        "DM:"
    )
    try:
        resp = _model.generate_content([system_prompt, prompt])
        text = resp.text.strip()
        return text
    except Exception:
        return None


