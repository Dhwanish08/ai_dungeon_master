import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default=os.path.join("data", "processed", "monster_behavior_samples.csv"))
	parser.add_argument("--out_dir", default=os.path.join("reports"))
	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	artifacts_dir = os.path.join(args.out_dir, "artifacts")
	os.makedirs(artifacts_dir, exist_ok=True)

	df = pd.read_csv(args.data)
	
	# Features: boss_hp, player_avg_hp, turn, location (encoded), player_intent (encoded)
	# Target: monster_behavior
	
	# Encode categoricals
	le_location = LabelEncoder()
	le_intent = LabelEncoder()
	le_behavior = LabelEncoder()
	
	df["location_encoded"] = le_location.fit_transform(df["location"])
	df["intent_encoded"] = le_intent.fit_transform(df["player_intent"])
	df["behavior_encoded"] = le_behavior.fit_transform(df["monster_behavior"])
	
	X = df[["boss_hp", "player_avg_hp", "turn", "location_encoded", "intent_encoded"]]
	y = df["behavior_encoded"]
	
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)
	
	# Preprocessing pipeline
	preprocess = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])
	
	# Models to compare
	models = {
		"LogReg": (LogisticRegression(max_iter=1000, class_weight="balanced"), {"clf__C": [0.5, 1.0, 2.0, 5.0]}),
		"RandomForest": (RandomForestClassifier(class_weight="balanced", random_state=42), {"clf__max_depth": [8, 12, None], "clf__min_samples_leaf": [1, 2]}),
		"GradientBoosting": (GradientBoostingClassifier(random_state=42), {"clf__learning_rate": [0.1, 0.2], "clf__max_depth": [2, 3], "clf__n_estimators": [100, 200]}),
		"LinearSVC+Calibrated": (CalibratedClassifierCV(LinearSVC(class_weight="balanced")), {}),
	}
	
	best_name, best_score, best_model = None, -1.0, None
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	
	for name, (est, grid) in models.items():
		pipe = Pipeline(steps=[("prep", preprocess), ("clf", est)])
		if grid:
			gs = GridSearchCV(pipe, grid, cv=3, n_jobs=-1, scoring="accuracy")
			gs.fit(X_train, y_train)
			best_est = gs.best_estimator_
		else:
			best_est = pipe
			best_est.fit(X_train, y_train)
		cv_scores = cross_val_score(best_est, X_train, y_train, cv=cv, scoring="accuracy")
		mean_cv = float(np.mean(cv_scores))
		print(f"{name}: CV mean acc={mean_cv:.4f}")
		if mean_cv > best_score:
			best_name, best_score, best_model = name, mean_cv, best_est
	
	# Fit best on full train, eval on test
	best_model.fit(X_train, y_train)
	y_pred = best_model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, zero_division=0, output_dict=True, labels=sorted(y_test.unique()), target_names=le_behavior.inverse_transform(sorted(y_test.unique())))
	
	# Save model with encoders
	model_path = os.path.join(artifacts_dir, "monster_behavior_model.joblib")
	joblib.dump({
		"model": best_model,
		"le_location": le_location,
		"le_intent": le_intent,
		"le_behavior": le_behavior,
	}, model_path)
	
	# Save metrics
	labels = le_behavior.inverse_transform(sorted(y_test.unique()))
	cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
	metrics = {
		"best_model": best_name,
		"cv_mean_acc": float(best_score),
		"test_acc": float(acc),
		"classification_report": report,
		"confusion_matrix": cm.tolist(),
		"labels": labels.tolist(),
		"artifacts": {
			"model": model_path,
		},
	}
	with open(os.path.join(args.out_dir, "metrics_monster_behavior.json"), "w") as f:
		json.dump(metrics, f, indent=2)
	
	print(f"Saved model to {model_path}")
	print(f"Saved metrics to {os.path.join(args.out_dir, 'metrics_monster_behavior.json')}")


if __name__ == "__main__":
	main()

