# ===============================================
# Clean → Encode → Classify (KNN) — Simplified
# ===============================================
# Dataset: Kaggle "Mushroom Classification"
# Expected file: ./mushrooms.csv

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer  # used ONCE as an example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 0) Load data
# -----------------------------
csv_path = "./mushroom.csv"   # TODO: place the file or change path
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path} — download from Kaggle and try again.")

df = pd.read_csv(csv_path)
print("Initial shape:", df.shape)

# Get the top 5 rows
df.head()

# -----------------------------
# 1) Basic cleaning (duplicates, empty rows)
# -----------------------------
before = df.shape[0]
# Remove duplicates
df.drop_duplicates(inplace=True)
print("Removed duplicates:", before - df.shape[0])

before = df.shape[0]
# Drop fully empty rows
df.dropna(how='all', inplace=True)
print("Dropped fully empty rows:", before - df.shape[0])

print("\nMissing values per column (top 10):")
# Display missing values per column
print(df.isnull().sum().head(10))

# -----------------------------
# 2) Target & split
# -----------------------------
TARGET_COL = "class"
if TARGET_COL not in df.columns:
    raise ValueError("Update TARGET_COL to match your dataset's label column.")

# Drop the target column from the "X"
X = df.drop(columns=[TARGET_COL])
# Set up the target column
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain/Test shapes:", X_train.shape, X_test.shape)

# Detect types BEFORE imputation/encoding
orig_numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
orig_categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
print("Original numeric cols:", orig_numeric_cols if orig_numeric_cols else "None")
print("Original categorical cols (sample):", orig_categorical_cols[:10])

# --------------------------------------------
# 3) IMPUTATION (MANUAL) + one sklearn example
# --------------------------------------------
X_train_imp = X_train.copy()
X_test_imp  = X_test.copy()

# a) Numeric → manual median
for col in orig_numeric_cols:
    median_val = X_train_imp[col].median()
    X_train_imp[col] = X_train_imp[col].fillna(median_val)
    X_test_imp[col]  = X_test_imp[col].fillna(median_val)

# b) Categorical → manual mode
for col in orig_categorical_cols:
    mode_val = X_train_imp[col].mode()[0]
    X_train_imp[col] = X_train_imp[col].fillna(mode_val)
    X_test_imp[col]  = X_test_imp[col].fillna(mode_val)

# c) One sklearn SimpleImputer example
IMPUTE_EXAMPLE_COL = "odor"
if IMPUTE_EXAMPLE_COL in X_train_imp.columns:
    cat_imp = SimpleImputer(strategy="most_frequent")
    # The fix: flatten (ravel) the 2D array returned by fit_transform()
    X_train_imp[IMPUTE_EXAMPLE_COL] = cat_imp.fit_transform(X_train_imp[[IMPUTE_EXAMPLE_COL]]).ravel()
    X_test_imp[IMPUTE_EXAMPLE_COL]  = cat_imp.transform(X_test_imp[[IMPUTE_EXAMPLE_COL]]).ravel()
    print(f"\nUsed sklearn SimpleImputer on column: {IMPUTE_EXAMPLE_COL}")
else:
    print(f"\n[Note] Example imputer column '{IMPUTE_EXAMPLE_COL}' not found. Skipping sklearn example.")

# --------------------------------------------
# 4) ENCODING
# --------------------------------------------
EXPLICIT_OHE_COL = "odor"

X_train_enc = X_train_imp.copy()
X_test_enc  = X_test_imp.copy()

if EXPLICIT_OHE_COL in X_train_enc.columns:
    # Explicit OHE on one column
    train_ohe = pd.get_dummies(X_train_enc[EXPLICIT_OHE_COL], prefix=EXPLICIT_OHE_COL)
    test_ohe = pd.get_dummies(X_test_enc[EXPLICIT_OHE_COL], prefix=EXPLICIT_OHE_COL)
    test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

    X_train_enc = pd.concat([X_train_enc.drop(columns=[EXPLICIT_OHE_COL]), train_ohe], axis=1)
    X_test_enc  = pd.concat([X_test_enc.drop(columns=[EXPLICIT_OHE_COL]),  test_ohe], axis=1)

    print(f"\nExplicit OHE applied to column: {EXPLICIT_OHE_COL}")
else:
    print(f"\n[Note] Explicit OHE column '{EXPLICIT_OHE_COL}' not found. Skipping explicit OHE step.")

# Remaining categoricals
remaining_cats = X_train_enc.select_dtypes(exclude=np.number).columns.tolist()

# Apply get_dummies
X_train_enc = pd.get_dummies(X_train_enc, columns=remaining_cats, drop_first=False)
X_test_enc  = pd.get_dummies(X_test_enc, columns=remaining_cats, drop_first=False)

# Align test to train columns
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

print("Encoded train shape:", X_train_enc.shape)
print("Encoded test shape:", X_test_enc.shape)

# --------------------------------------------
# 5) NORMALIZE numeric columns (if any)
# --------------------------------------------
numeric_cols_to_scale = [c for c in orig_numeric_cols if c in X_train_enc.columns]

scaler = StandardScaler()
if numeric_cols_to_scale:
    X_train_enc[numeric_cols_to_scale] = scaler.fit_transform(X_train_enc[numeric_cols_to_scale])
    X_test_enc[numeric_cols_to_scale]  = scaler.transform(X_test_enc[numeric_cols_to_scale])
    print("Scaled columns:", numeric_cols_to_scale)
else:
    print("No numeric columns to scale.")

# --------------------------------------------
# 6) KNN + Grid Search
# --------------------------------------------
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2]  # 1=Manhattan, 2=Euclidean
}

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_enc, y_train)

best_knn = grid.best_estimator_
print("\nBest estimator:", best_knn)
print("Best params:", grid.best_params_)

# --------------------------------------------
# 7) Evaluate on held-out TEST
# --------------------------------------------
y_pred = best_knn.predict(X_test_enc)

test_acc = accuracy_score(y_test, y_pred)
print("\nTest accuracy: {:.4f}".format(test_acc))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
plt.figure()
disp.plot(values_format="d")
plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 8) Validation curve (CV accuracy vs k)
# --------------------------------------------
cvres = pd.DataFrame(grid.cv_results_)
plotdf = cvres[["param_n_neighbors", "param_weights", "param_p", "mean_test_score"]].rename(
    columns={"param_n_neighbors":"k", "param_weights":"weights", "param_p":"p"}
)

plt.figure()
for (w, pval), sub in plotdf.groupby(["weights", "p"]):
    sub = sub.sort_values("k")
    plt.plot(sub["k"], sub["mean_test_score"], marker="o", label=f"weights={w}, p={pval}")
plt.xlabel("k (n_neighbors)")
plt.ylabel("Mean CV Accuracy")
plt.title("CV Accuracy vs k (by weights & p)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------
# 9) ROC curve
# --------------------------------------------
if hasattr(best_knn, "predict_proba"):
    classes_ = np.unique(y_train)
    y_test_enc_int = pd.Categorical(y_test, categories=classes_).codes
    proba = best_knn.predict_proba(X_test_enc)
    print(np.unique(proba)) 


    if len(classes_) == 2:
        fpr, tpr, _ = roc_curve(y_test_enc_int, proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Test)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    else:
        y_bin = label_binarize(y_test_enc_int, classes=range(len(classes_)))
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"Micro-average AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Micro-average)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

# --------------------------------------------
# 10) Print summary of features
# --------------------------------------------
print(f"\nFinal feature count: {X_train_enc.shape[1]}")
print(f"Best KNN: {best_knn}")
