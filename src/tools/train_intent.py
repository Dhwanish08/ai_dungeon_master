import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default=os.path.join("data", "processed", "player_intent_samples.csv"))
	parser.add_argument("--out_dir", default=os.path.join("reports"))
	args = parser.parse_args()

	df = pd.read_csv(args.data)
	if "text" not in df.columns or "intent" not in df.columns:
		raise ValueError("Dataset must contain 'text' and 'intent' columns.")

	X = df["text"].astype(str)
	y = df["intent"].astype(str)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	classes = np.unique(y_train)
	class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
	weight_dict = {cls: w for cls, w in zip(classes, class_weights)}

	pipe = Pipeline(
		steps=[
			("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")),
			("clf", LogisticRegression(max_iter=2000, class_weight=weight_dict)),
		]
	)

	param_grid = {
		"clf__C": [0.5, 1.0, 2.0, 3.0],
		"tfidf__min_df": [1, 2],
	}

	n_splits = min(5, y_train.nunique())
	cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
	grid.fit(X_train, y_train)

	best_model = grid.best_estimator_
	y_pred = best_model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

	out_dir = args.out_dir
	os.makedirs(out_dir, exist_ok=True)
	artifacts_dir = os.path.join(out_dir, "artifacts")
	os.makedirs(artifacts_dir, exist_ok=True)
	model_path = os.path.join(artifacts_dir, "intent_model.joblib")
	joblib.dump(best_model, model_path)

	metrics = {
		"task": "player_intent",
		"best_params": grid.best_params_,
		"cv_best_score": float(grid.best_score_),
		"test_accuracy": float(acc),
		"classification_report": report,
		"artifacts": {"model": model_path},
	}
	with open(os.path.join(out_dir, "metrics_intent.json"), "w") as f:
		json.dump(metrics, f, indent=2)

	print(f"Saved intent model to {model_path}")
	print(f"Saved metrics to {os.path.join(out_dir, 'metrics_intent.json')}")


if __name__ == "__main__":
	main()


