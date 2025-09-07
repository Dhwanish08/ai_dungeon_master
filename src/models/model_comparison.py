# src/models/model_comparison.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# =============================
# 1. Load Data
# =============================
df = pd.read_csv("data/processed/Dd5e_monsters_clean.csv")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Available columns:", df.columns.tolist())
print(df.head())

# =============================
# 2. Preprocess Target (Alignment)
# =============================
def simplify_alignment(text):
    if pd.isna(text):
        return "unaligned"
    text = text.lower()
    if "chaotic evil" in text: return "chaotic evil"
    if "lawful evil" in text: return "lawful evil"
    if "neutral evil" in text: return "neutral evil"
    if "lawful good" in text: return "lawful good"
    if "chaotic good" in text: return "chaotic good"
    if "neutral good" in text: return "neutral good"
    if "neutral" in text: return "neutral"
    if "unaligned" in text: return "unaligned"
    return text.strip()

df["alignment_simple"] = df["alignment"].apply(simplify_alignment)

print("\nUnique simplified alignments:", df["alignment_simple"].value_counts())

# =============================
# 2b. Handle Rare Classes
# =============================
min_class_size = 2
alignment_counts = df["alignment_simple"].value_counts()
rare_classes = alignment_counts[alignment_counts < min_class_size].index

df["alignment_simple"] = df["alignment_simple"].replace(rare_classes, "other")

print("\nAfter merging rare classes:")
print(df["alignment_simple"].value_counts())


# =============================
# 3. Feature Engineering
# =============================
# Extract numeric challenge rating from string (e.g., "1/4 (50 XP)" -> 0.25)
def parse_cr(cr):
    if pd.isna(cr):
        return np.nan
    cr = str(cr).split()[0]
    if "/" in cr:
        num, denom = cr.split("/")
        return float(num) / float(denom)
    try:
        return float(cr)
    except:
        return np.nan

df["challenge_rating_num"] = df["challenge_rating"].apply(parse_cr)

# Define features + target
X = df[["size", "hit_points", "armor_class", "challenge_rating_num", "alignment_simple"]].copy()
y = df["alignment_simple"]

# =============================
# 4. Train/Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# 5. Preprocessing Pipelines
# =============================
numeric_features = ["hit_points", "armor_class", "challenge_rating_num"]
categorical_features = ["size"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =============================
# 6. Models to Compare
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "SVM": SVC(kernel="rbf", class_weight="balanced")
}

# =============================
# 7. Train + Evaluate
# =============================
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
