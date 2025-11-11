import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def parse_cr(cr):
    if pd.isna(cr):
        return np.nan
    cr = str(cr).split()[0]
    if "/" in cr:
        num, denom = cr.split("/")
        try:
            return float(num) / float(denom)
        except Exception:
            return np.nan
    try:
        return float(cr)
    except Exception:
        return np.nan


def collapse5(t: str) -> str:
    if pd.isna(t):
        return "unaligned"
    tt = str(t).lower()
    if "good" in tt:
        return "good"
    if "evil" in tt:
        return "evil"
    if "unaligned" in tt:
        return "unaligned"
    return "neutral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "processed", "Dd5e_monsters_clean.csv"))
    parser.add_argument("--out_dir", default=os.path.join("reports"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    figs_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip().str.lower()
    df["challenge_rating_num"] = df["challenge_rating"].apply(parse_cr)
    for col in ["hit_points", "armor_class"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["y5"] = df["alignment"].apply(collapse5)

    X = df[["name", "size", "hit_points", "armor_class", "challenge_rating_num"]]
    y = df["y5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english"), "name"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["size"]),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), ["hit_points", "armor_class", "challenge_rating_num"]),
        ]
    )

    # Compare SVC and LogReg via small grids
    models = {
        "LinearSVC": (LinearSVC(class_weight="balanced"), {"clf__C": [0.5, 1.0, 2.0, 5.0]}),
        "LogReg": (LogisticRegression(max_iter=1000, class_weight="balanced"), {"clf__C": [0.2, 0.5, 1.0, 2.0, 5.0]})
    }

    best_name, best_score, best_model = None, -1.0, None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (est, grid) in models.items():
        pipe = Pipeline(steps=[("prep", preprocess), ("clf", est)])
        gs = GridSearchCV(pipe, grid, cv=3, n_jobs=-1, scoring="accuracy")
        gs.fit(X_train, y_train)
        cv_scores = cross_val_score(gs.best_estimator_, X_train, y_train, cv=cv, scoring="accuracy")
        mean_cv = float(np.mean(cv_scores))
        print(f"{name}: CV mean acc={mean_cv:.4f}, best={gs.best_params_}")
        if mean_cv > best_score:
            best_name, best_score, best_model = name, mean_cv, gs.best_estimator_

    # Fit best on train, eval on test
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Save artifacts
    artifacts_dir = os.path.join(args.out_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "alignment_model.joblib")
    joblib.dump(best_model, model_path)

    # Confusion matrix plot
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Collapsed 5)")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(labels)), labels=labels)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(figs_dir, "alignment_confusion_5classes.png")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics JSON
    metrics = {
        "best_model": best_name,
        "cv_mean_acc": best_score,
        "test_acc": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "artifacts": {
            "model": model_path,
            "confusion_png": cm_path,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()


