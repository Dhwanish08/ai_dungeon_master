# 🧙‍♂️ AI Dungeon Master (Data + EDA + Baseline Models)

## Streamlit Hybrid Demo (Local ML + Gemini Narrator)

This repo includes a simple Streamlit demo that showcases a hybrid architecture:
- Local model and rules handle intents and monster behavior (deterministic outcomes).
- Gemini (or a deterministic fallback) acts as narrator only and never decides outcomes.

How to run:
- Create and activate your Python environment (see requirements.txt).
- Optionally export your Gemini API key: `export GEMINI_API_KEY=YOUR_KEY`
- Start the app:
  - `streamlit run src/ui/streamlit_app.py`
  - or `bash src/ui/run_demo.sh`

UI notes:
- Start screen lets you select number of players, enter player names, and begin.
- Main screen shows a dark, futuristic theme with a scrollable story log and action input.
- Top section displays the current scene/location and turn.
- Sidebar shows party status, world location, dice log, and a panel of local model outputs (intent + monster behavior).
- Quick actions: Attack / Explore / Talk / Inventory / Run to guide testers.

Gemini fallback:
- If `GEMINI_API_KEY` is not set or the API call fails, the app uses a deterministic narrator (`src/ui/gemini_fallback.py`).

Honest description:
- Hybrid prototype: local model = rules/intent/monster; Gemini = narrator only.

This repo currently focuses on exploring D&D 5e datasets (monsters, spells), cleaning them, generating EDA plots, and training simple baseline models (no RL or game engine yet).

## 📂 Project Structure
ai_dungeon_master/
│── data/
│   ├── raw/            # original datasets (input)
│   └── processed/      # cleaned CSVs and generated figures (output)
│── notebooks/          # Jupyter exploration
│── reports/
│   └── figures/        # finalized figures for reports
│── src/
│   ├── data/
│   │   └── clean_data.py              # produces cleaned CSVs
│   ├── eda/
│   │   └── eda_monsters.py            # generates monster EDA plots
│   └── models/
│       ├── monster_alignment_model.py # TF-IDF + Logistic Regression
│       └── model_comparison.py        # Compares classic ML models
│── requirements.txt
│── README.md

## ⚙️ Setup
1) Create a virtual environment and install deps:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Ensure raw data files exist:
- `data/raw/Dd5e_monsters.csv`
- `data/raw/dnd-spells.csv`

## 🚀 How to Run
1) Clean datasets (writes to `data/processed/`):
```bash
python src/data/clean_data.py
```

2) Generate EDA plots (writes to `reports/figures/`):
```bash
python src/eda/eda_monsters.py
```

3) Train/evaluate models:
```bash
python src/models/monster_alignment_model.py
python src/models/model_comparison.py
```

Or run everything with one command:
```bash
python main.py all
```

### Alignment model CLI (exported)
- Train and export best model (collapsed 5 classes), save metrics and confusion matrix:
```bash
python src/tools/train_alignment.py --data data/processed/Dd5e_monsters_clean.csv --out_dir reports
```
- Predict with exported model:
```bash
python src/tools/predict_alignment.py "Adult Black Dragon" --size Huge --hp 256 --ac 19 --cr 14
```

## 🧙 RPG AI Dungeon Master (MVP)
Run a local, no-API rule-based DM you can play in the terminal:
```bash
python main.py play
```

Train a small PPO agent in the simplified DM environment (local CPU/GPU):
```bash
python main.py train
```
This uses `gymnasium` + `stable-baselines3` and saves a model `dm_ppo.zip`.

### 🖥️ Web UI (Streamlit)
Play in a simple browser UI (no API):
```bash
streamlit run src/ui/app.py
# or
python main.py ui
```

Optional: Enable Gemini for richer narration
- Set `GEMINI_API_KEY` in your environment. If present, the DM will use Gemini for story text, guided by your ML model outputs (intent + predicted alignment). Without the key, it falls back to local rule-based narration.

Artifacts generated:
- EDA figures: `reports/figures/*.png`
- Executed demo notebook: `notebooks/final_demo_executed.ipynb`
- Demo HTML report: `reports/final_demo.html`
- Model metrics JSON: `reports/metrics.json`
- PPO rewards curve: `reports/figures/ppo_rewards.png` (after `python main.py train`)

Goal: Predict monster alignment from biological + stat traits and lore text.
Approach:
- Data ingestion & cleaning
- EDA & domain exploration (monster ecology, CR, alignment trends)
- Feature engineering (numeric + categorical + text TF-IDF)
- Train logistic regression, SVM, random forest
- Compare models, report metrics

## ✅ What’s Implemented
- Data cleaning for monsters and spells to consistent column names/types.
- EDA: alignment/size distributions, armor class and HP analyses, challenge rating.
- Baseline models for predicting simplified alignment categories.

## 🧭 Notes
- The original README mentioned RL/Transformers/game engine, but those are not present yet.
- A `.gitignore` is included to avoid committing `venv/` and generated artifacts.