import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(description: str, cmd: list[str]) -> None:
    print(f"\n=== {description} ===")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def ensure_raw_data() -> None:
    required = [
        os.path.join(REPO_ROOT, "data", "raw", "Dd5e_monsters.csv"),
        os.path.join(REPO_ROOT, "data", "raw", "dnd-spells.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing required raw data files:")
        for p in missing:
            print(" -", os.path.relpath(p, REPO_ROOT))
        sys.exit(1)


def cmd_clean() -> None:
    ensure_raw_data()
    run_step(
        "Clean datasets",
        [sys.executable, os.path.join(REPO_ROOT, "src", "data", "clean_data.py")],
    )


def cmd_eda() -> None:
    run_step(
        "Generate EDA plots",
        [sys.executable, os.path.join(REPO_ROOT, "src", "eda", "eda_monsters.py")],
    )


def cmd_models() -> None:
    run_step(
        "Train TF-IDF Logistic Regression baseline",
        [sys.executable, os.path.join(REPO_ROOT, "src", "models", "monster_alignment_model.py")],
    )
    run_step(
        "Compare classic ML models (LR, RF, SVM)",
        [sys.executable, os.path.join(REPO_ROOT, "src", "models", "model_comparison.py")],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict monster alignment from biological/stat traits and lore text.\n"
            "Pipeline: clean -> EDA -> models."
        )
    )
    parser.add_argument(
        "command",
        choices=["clean", "eda", "models", "all", "play", "train", "ui"],
        help="Which step to run",
    )
    args = parser.parse_args()

    if args.command == "clean":
        cmd_clean()
    elif args.command == "eda":
        cmd_eda()
    elif args.command == "models":
        cmd_models()
    elif args.command == "all":
        cmd_clean()
        cmd_eda()
        cmd_models()
    elif args.command == "play":
        from src.game.engine import play_loop
        play_loop()
    elif args.command == "train":
        # short PPO training run as a placeholder
        from src.rl.train_ppo import train_ppo
        train_ppo(total_timesteps=5000)
    elif args.command == "ui":
        # Launch Streamlit UI
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.join(REPO_ROOT, "src", "ui", "app.py")])


if __name__ == "__main__":
    main()


