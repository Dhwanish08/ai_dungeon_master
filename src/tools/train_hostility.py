import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def parse_cr(cr):
	if pd.isna(cr):
		return np.nan
	cr = str(cr).split()[0]
	if "/" in cr:
		try:
			num, denom = cr.split("/")
			return float(num) / float(denom)
		except Exception:
			return np.nan
	try:
		return float(cr)
	except Exception:
		return np.nan


def make_hostility(alignment: str) -> str:
	if pd.isna(alignment):
		return "non_hostile"
	a = str(alignment).lower()
	if "evil" in a:
		return "hostile"
	# Treat others as non-hostile for clearer separability
	return "non_hostile"


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default=os.path.join("data", "processed", "Dd5e_monsters_clean.csv"))
	parser.add_argument("--out_dir", default=os.path.join("reports"))
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	artifacts_dir = os.path.join(args.out_dir, "artifacts")
	os.makedirs(artifacts_dir, exist_ok=True)

	df = pd.read_csv(args.data)
	df.columns = df.columns.str.strip().str.lower()
	df["challenge_rating_num"] = df["challenge_rating"].apply(parse_cr)
	for col in ["hit_points", "armor_class"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df["hostility"] = df["alignment"].apply(make_hostility)

	# Build robust feature set: numeric + categorical + combined text from available columns
	text_candidates = [c for c in ["name", "type", "subtype", "languages", "condition_immunities", "damage_immunities", "damage_resistances", "senses"] if c in df.columns]
	if not text_candidates:
		text_candidates = ["name"]
	df["__text__"] = (
		df[text_candidates].fillna("").astype(str).agg(" ".join, axis=1)
	)

	X = df[["__text__", "size", "hit_points", "armor_class", "challenge_rating_num"]]
	y = df["hostility"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	preprocess = ColumnTransformer(
		transformers=[
			("text", TfidfVectorizer(max_features=12000, ngram_range=(1, 3), stop_words="english", min_df=2), "__text__"),
			("cat", OneHotEncoder(handle_unknown="ignore"), ["size"]),
			("num", Pipeline(steps=[
				("imputer", SimpleImputer(strategy="median")),
				("scaler", StandardScaler()),
			]), ["hit_points", "armor_class", "challenge_rating_num"]),
		]
	)

	models = {
		"LinearSVC+Calibrated": (
			CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
			{}  # wrapped model doesn't expose C easily for grid; keep default for stability
		),
		"LogReg": (
			LogisticRegression(max_iter=2000, class_weight="balanced"),
			{"clf__C": [0.5, 1.0, 2.0, 5.0, 10.0]}
		),
		"RandomForest": (
			RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=42),
			{"clf__max_depth": [None, 12, 20], "clf__min_samples_leaf": [1, 2, 4]}
		),
		"GradientBoosting": (
			GradientBoostingClassifier(random_state=42),
			{"clf__n_estimators": [200, 300], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}
		),
		"SoftVoting": (
			VotingClassifier(
				estimators=[
					("svc", CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
					("gb", GradientBoostingClassifier(random_state=42, n_estimators=250, learning_rate=0.1, max_depth=3)),
					("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
				],
				voting="soft",
				weights=[2, 2, 1],
				n_jobs=None,
			),
			{
				# try a couple of weightings
				"clf__weights": [(2,2,1), (3,2,1)]
			}
		),
	}

	best_name, best_score, best_model = None, -1.0, None
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

	for name, (est, grid) in models.items():
		pipe = Pipeline(steps=[("prep", preprocess), ("clf", est)])
		gs = GridSearchCV(pipe, grid if grid else {"clf": [est]}, cv=3, n_jobs=-1, scoring="accuracy")
		gs.fit(X_train, y_train)
		# Evaluate via 5-fold CV using best params
		cv_scores = []
		for tr_idx, val_idx in cv.split(X_train, y_train):
			X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
			y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
			model = gs.best_estimator_
			model.fit(X_tr, y_tr)
			pred = model.predict(X_val)
			cv_scores.append(accuracy_score(y_val, pred))
		mean_cv = float(np.mean(cv_scores))
		print(f"{name}: CV mean acc={mean_cv:.4f}, best={gs.best_params_}")
		if mean_cv > best_score:
			best_name, best_score, best_model = name, mean_cv, gs.best_estimator_

	best_model.fit(X_train, y_train)
	y_pred = best_model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

	model_path = os.path.join(artifacts_dir, "hostility_model.joblib")
	joblib.dump(best_model, model_path)

	metrics = {
		"task": "hostility_binary",
		"best_model": best_name,
		"cv_mean_acc": best_score,
		"test_acc": float(acc),
		"classification_report": report,
		"artifacts": {"model": model_path},
	}
	with open(os.path.join(args.out_dir, "metrics_hostility.json"), "w") as f:
		json.dump(metrics, f, indent=2)

	print(f"Saved hostility model to {model_path}")
	print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics_hostility.json')}")


if __name__ == "__main__":
	main()


