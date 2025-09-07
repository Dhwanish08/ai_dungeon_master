# src/models/monster_alignment_model.py

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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

def clean_alignment(text):
    if pd.isna(text):
        return "unknown"
    # Keep only alignment part (after comma if exists)
    parts = str(text).split(",")
    if len(parts) > 1:
        return parts[-1].strip().lower()
    return str(text).strip().lower()

df["alignment"] = df["alignment"].apply(clean_alignment)

print("\nUnique alignments:", df["alignment"].nunique())
print(df["alignment"].value_counts().head())

# 3. Feature Engineering
df["text_features"] = (
    df["name"].astype(str) + " " +
    df["size"].astype(str) + " " +
    df["armor_class"].astype(str)
)


X = df["text_features"]
y = df["alignment"]

# Remove alignment classes with <2 samples
alignment_counts = y.value_counts()
valid_alignments = alignment_counts[alignment_counts >= 2].index
mask = y.isin(valid_alignments)
X = X[mask]
y = y[mask]

# =============================
# 4. Encode Labels
# =============================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# =============================
# 5. Train/Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =============================
# 6. Vectorize Text
# =============================
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =============================
# 7. Train Model
# =============================
model = LogisticRegression(max_iter=200, class_weight="balanced")
model.fit(X_train_vec, y_train)

# =============================
# 8. Evaluate
# =============================
y_pred = model.predict(X_test_vec)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Fix: pass the actual labels seen in test set
labels_in_test = np.unique(y_test)
target_names = le.inverse_transform(labels_in_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names))
