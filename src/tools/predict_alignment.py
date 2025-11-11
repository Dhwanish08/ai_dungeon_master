import os
import argparse
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Monster name text (used for TF-IDF)")
    parser.add_argument("--size", default="Medium")
    parser.add_argument("--hp", type=float, default=float('nan'))
    parser.add_argument("--ac", type=float, default=float('nan'))
    parser.add_argument("--cr", type=float, default=float('nan'))
    parser.add_argument("--model", default=os.path.join("reports", "artifacts", "alignment_model.joblib"))
    args = parser.parse_args()

    model = joblib.load(args.model)
    X = pd.DataFrame([
        {
            "name": args.name,
            "size": args.size,
            "hit_points": args.hp,
            "armor_class": args.ac,
            "challenge_rating_num": args.cr,
        }
    ])
    pred = model.predict(X)[0]
    print(pred)


if __name__ == "__main__":
    main()


