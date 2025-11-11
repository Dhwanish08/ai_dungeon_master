import os
from typing import Optional

try:
    import joblib
    import pandas as pd
except Exception:
    joblib = None
    pd = None


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports", "artifacts", "alignment_model.joblib")
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if joblib is None:
        return None
    if os.path.exists(MODEL_PATH):
        try:
            _MODEL = joblib.load(MODEL_PATH)
            return _MODEL
        except Exception:
            return None
    return None


def predict_alignment(name: str, size: str, hp: Optional[float], ac: Optional[float], cr: Optional[float]) -> Optional[str]:
    model = _load_model()
    if model is None or pd is None:
        return None
    X = pd.DataFrame([
        {
            "name": name,
            "size": size,
            "hit_points": hp,
            "armor_class": ac,
            "challenge_rating_num": cr,
        }
    ])
    try:
        pred = model.predict(X)[0]
        return str(pred)
    except Exception:
        return None


