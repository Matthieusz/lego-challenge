"""
LEGO Color Classification Pipeline.

Loads pixel-level data from archive/legocolor-extended.csv,
trains and evaluates color classifiers using group-stratified splits
(by Photo_number) to avoid data leakage.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    HalvingGridSearchCV,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("archive/legocolor-extended.csv")
DELIMITER = ";"
FEATURE_COLS = [
    "R",
    "G",
    "B",
    "Hue",
    "Saturation",
    "Value",
    "Y",
    "U",
    "V",
    "R_Average",
    "G_Average",
    "B_Average",
]
TARGET_COL = "Color"
GROUP_COL = "Photo_number"
TEST_SIZE = 0.2
RANDOM_STATE = 42
PLOT_DIR = Path("plots")
CV_FOLDS = 5


def load_data(path: Path, delimiter: str) -> pd.DataFrame:
    """Load and validate the dataset."""
    df = pd.read_csv(path, delimiter=delimiter)
    missing = [c for c in FEATURE_COLS + [TARGET_COL, GROUP_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def group_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split grouped by Photo_number (no leakage)."""
    groups = df[GROUP_COL].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def extract_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, groups arrays from a DataFrame."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    groups = df[GROUP_COL].values
    return X, y, groups


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Model training functions (explicit, one per model)
# ---------------------------------------------------------------------------


def train_knn(X_train: np.ndarray, y_train: np.ndarray) -> KNeighborsClassifier:
    """Train a k-NN baseline (distance-based, requires scaling)."""
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> RandomForestClassifier:
    """Train a tuned Random Forest using GroupKFold CV."""
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [None, 20, 30],
        "min_samples_leaf": [1, 2],
    }
    cv = GroupKFold(n_splits=CV_FOLDS)
    grid = GridSearchCV(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups)
    print(f"  Best RF params: {grid.best_params_}")
    print(f"  Best CV F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> SVC:
    """Train a tuned SVM using GroupKFold CV."""
    param_grid = [
        {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
    ]
    cv = GroupKFold(n_splits=CV_FOLDS)
    grid = GridSearchCV(
        SVC(class_weight="balanced", random_state=RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups)
    print(f"  Best SVM params: {grid.best_params_}")
    print(f"  Best CV F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_hgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> HistGradientBoostingClassifier:
    """Train a tuned HistGradientBoosting using GroupKFold CV."""
    param_grid = {
        "learning_rate": [0.01, 0.1],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [1, 5],
    }
    cv = GroupKFold(n_splits=CV_FOLDS)
    grid = GridSearchCV(
        HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups)
    print(f"  Best HGB params: {grid.best_params_}")
    print(f"  Best CV F1 (macro): {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_logreg(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a multinomial Logistic Regression baseline (no CV grid)."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> MLPClassifier:
    """Train a tuned MLP using successive halving GroupKFold CV."""
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,)],
        "alpha": [0.0001, 0.001],
    }
    cv = GroupKFold(n_splits=CV_FOLDS)
    halving = HalvingGridSearchCV(
        MLPClassifier(
            solver="adam",
            activation="relu",
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        factor=3,
    )
    halving.fit(X_train, y_train, groups=groups)
    print(f"  Best MLP params: {halving.best_params_}")
    print(f"  Best CV F1 (macro): {halving.best_score_:.4f}")
    return halving.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation & plotting
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (macro): {macro_f1:.4f}")
    print("\n  Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=le.classes_,
            zero_division=0,
        )
    )
    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": macro_f1,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    filename: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Saved confusion matrix to {filename}")


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    filename: Path,
) -> None:
    """Save a horizontal bar chart of feature importances."""
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(importances)), importances[indices], align="center")
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"  Saved feature importances to {filename}")


def print_summary_table(results: list[dict]) -> None:
    """Print a compact comparison table."""
    print(f"\n{'=' * 44}")
    print("  Model Comparison")
    print(f"{'=' * 44}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'F1 (macro)':>10}")
    print(f"  {'-' * 42}")
    for r in results:
        print(f"  {r['name']:<22} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")
    print(f"{'=' * 44}")


# ---------------------------------------------------------------------------
# Two-stage classifier
# ---------------------------------------------------------------------------

GREY_CLASSES = {"Black", "Dark Bluish Grey", "Light Bluish Grey"}


def train_two_stage_mlp(
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    train_groups: np.ndarray,
) -> tuple[MLPClassifier, MLPClassifier, MLPClassifier, LabelEncoder, LabelEncoder]:
    """Train a two-stage MLP: (1) grey vs color, (2a) which grey, (2b) which color."""
    # Stage 1: binary grey vs color
    y_binary = np.array(["grey" if c in GREY_CLASSES else "color" for c in y_train_raw])
    le_binary = LabelEncoder()
    y_binary_enc = le_binary.fit_transform(y_binary)

    print("  Stage 1: training grey vs color classifier...")
    stage1 = train_mlp(X_train, y_binary_enc, train_groups)

    # Stage 2a: which grey (subset of training data)
    grey_mask = np.array([c in GREY_CLASSES for c in y_train_raw])
    X_grey = X_train[grey_mask]
    y_grey = y_train_raw[grey_mask]
    groups_grey = train_groups[grey_mask]
    le_grey = LabelEncoder()
    y_grey_enc = le_grey.fit_transform(y_grey)

    print("  Stage 2a: training grey-class classifier...")
    stage2_grey = train_mlp(X_grey, y_grey_enc, groups_grey)

    # Stage 2b: which color (subset of training data)
    color_mask = ~grey_mask
    X_color = X_train[color_mask]
    y_color = y_train_raw[color_mask]
    groups_color = train_groups[color_mask]
    le_color = LabelEncoder()
    y_color_enc = le_color.fit_transform(y_color)

    print("  Stage 2b: training color-class classifier...")
    stage2_color = train_mlp(X_color, y_color_enc, groups_color)

    return stage1, stage2_grey, stage2_color, le_binary, le_grey, le_color


def predict_two_stage(
    stage1: MLPClassifier,
    stage2_grey: MLPClassifier,
    stage2_color: MLPClassifier,
    le_binary: LabelEncoder,
    le_grey: LabelEncoder,
    le_color: LabelEncoder,
    X: np.ndarray,
) -> np.ndarray:
    """Run two-stage prediction on scaled features."""
    # Stage 1: grey vs color
    y_binary_pred = stage1.predict(X)
    y_binary_labels = le_binary.inverse_transform(y_binary_pred)

    predictions = np.empty(len(X), dtype=object)

    # Stage 2a: grey branch
    grey_mask = y_binary_labels == "grey"
    if grey_mask.any():
        y_grey_pred = stage2_grey.predict(X[grey_mask])
        predictions[grey_mask] = le_grey.inverse_transform(y_grey_pred)

    # Stage 2b: color branch
    color_mask = ~grey_mask
    if color_mask.any():
        y_color_pred = stage2_color.predict(X[color_mask])
        predictions[color_mask] = le_color.inverse_transform(y_color_pred)

    return predictions


def main() -> None:
    PLOT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH, DELIMITER)
    print(f"  Rows: {len(df)}, Colors: {df[TARGET_COL].nunique()}")

    # Group split: all pixels from the same photo stay in one split
    df_train, df_test = group_train_test_split(df, TEST_SIZE, RANDOM_STATE)
    print(f"  Train photos: {df_train[GROUP_COL].nunique()} ({len(df_train)} samples)")
    print(f"  Test photos : {df_test[GROUP_COL].nunique()} ({len(df_test)} samples)")

    X_train, y_train, train_groups = extract_features(df_train)
    X_test, y_test, _test_groups = extract_features(df_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Scale features (required for k-NN, SVM, LogReg, MLP; benign for tree models)
    X_train_s, X_test_s, _scaler = scale_features(X_train, X_test)

    results: list[dict] = []

    # --- Baseline: k-NN ---
    print("\nTraining k-NN baseline...")
    knn = train_knn(X_train_s, y_train_enc)
    res_knn = evaluate_model(knn, "k-NN (k=5)", X_test_s, y_test_enc, le)
    results.append(res_knn)

    # --- Primary: Random Forest ---
    print("\nTraining Random Forest (with GroupKFold CV)...")
    rf = train_random_forest(X_train_s, y_train_enc, train_groups)
    res_rf = evaluate_model(rf, "Random Forest", X_test_s, y_test_enc, le)
    results.append(res_rf)

    # --- SVM ---
    print("\nTraining SVM (with GroupKFold CV)...")
    svm = train_svm(X_train_s, y_train_enc, train_groups)
    res_svm = evaluate_model(svm, "SVM", X_test_s, y_test_enc, le)
    results.append(res_svm)

    # --- HistGradientBoosting ---
    print("\nTraining HistGradientBoosting (with GroupKFold CV)...")
    hgb = train_hgb(X_train_s, y_train_enc, train_groups)
    res_hgb = evaluate_model(hgb, "HistGradientBoosting", X_test_s, y_test_enc, le)
    results.append(res_hgb)

    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    logreg = train_logreg(X_train_s, y_train_enc)
    res_logreg = evaluate_model(logreg, "Logistic Regression", X_test_s, y_test_enc, le)
    results.append(res_logreg)

    # --- MLP ---
    print("\nTraining MLP (with HalvingGridSearchCV)...")
    mlp = train_mlp(X_train_s, y_train_enc, train_groups)
    res_mlp = evaluate_model(mlp, "MLP", X_test_s, y_test_enc, le)
    results.append(res_mlp)

    # --- Two-stage MLP ---
    print("\nTraining Two-Stage MLP...")
    (
        stage1,
        stage2_grey,
        stage2_color,
        le_binary,
        le_grey,
        le_color,
    ) = train_two_stage_mlp(X_train_s, y_train, train_groups)

    y_pred_ts = predict_two_stage(
        stage1, stage2_grey, stage2_color, le_binary, le_grey, le_color, X_test_s
    )
    # Encode string predictions back to le indices for consistent evaluation
    y_pred_ts_enc = le.transform(y_pred_ts)
    acc_ts = accuracy_score(y_test_enc, y_pred_ts_enc)
    macro_f1_ts = f1_score(y_test_enc, y_pred_ts_enc, average="macro", zero_division=0)

    print(f"\n{'=' * 50}")
    print("  Two-Stage MLP")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc_ts:.4f}")
    print(f"  F1 (macro): {macro_f1_ts:.4f}")
    print("\n  Classification report:")
    print(
        classification_report(
            y_test_enc,
            y_pred_ts_enc,
            target_names=le.classes_,
            zero_division=0,
        )
    )
    res_ts = {
        "name": "Two-Stage MLP",
        "accuracy": acc_ts,
        "f1_macro": macro_f1_ts,
        "y_pred": y_pred_ts_enc,
    }
    results.append(res_ts)

    # --- MLP + brightness-threshold post-processing for greys ---
    print("\nTraining MLP + Grey Threshold...")

    # Find optimal Y threshold for Dark Bluish Grey vs Light Bluish Grey
    # Use unscaled X_train so Y has physical meaning
    y_idx = FEATURE_COLS.index("Y")
    grey_mask_train = np.isin(y_train, ["Dark Bluish Grey", "Light Bluish Grey"])
    grey_Y = X_train[grey_mask_train, y_idx]
    grey_labels = y_train[grey_mask_train]

    # Search thresholds between min and max Y of grey samples
    thresholds = np.linspace(grey_Y.min(), grey_Y.max(), 200)
    best_thresh = None
    best_err = float("inf")
    for t in thresholds:
        pred = np.where(grey_Y > t, "Light Bluish Grey", "Dark Bluish Grey")
        err = np.sum(pred != grey_labels)
        if err < best_err:
            best_err = err
            best_thresh = t

    print(
        f"  Optimal Y threshold: {best_thresh:.1f} (training error: {best_err} / {len(grey_Y)})"
    )

    # MLP base predictions
    y_pred_mlp = mlp.predict(X_test_s)
    y_pred_mlp_labels = le.inverse_transform(y_pred_mlp)

    # Post-process: override grey predictions using Y threshold on unscaled test data
    grey_pred_mask = np.isin(
        y_pred_mlp_labels, ["Dark Bluish Grey", "Light Bluish Grey"]
    )
    grey_Y_test = X_test[grey_pred_mask, y_idx]
    corrected = np.where(
        grey_Y_test > best_thresh, "Light Bluish Grey", "Dark Bluish Grey"
    )
    y_pred_mlp_labels[grey_pred_mask] = corrected
    y_pred_thresh_enc = le.transform(y_pred_mlp_labels)

    acc_thresh = accuracy_score(y_test_enc, y_pred_thresh_enc)
    macro_f1_thresh = f1_score(
        y_test_enc, y_pred_thresh_enc, average="macro", zero_division=0
    )

    print(f"\n{'=' * 50}")
    print("  MLP + Grey Threshold")
    print(f"{'=' * 50}")
    print(f"  Accuracy : {acc_thresh:.4f}")
    print(f"  F1 (macro): {macro_f1_thresh:.4f}")
    print("\n  Classification report:")
    print(
        classification_report(
            y_test_enc,
            y_pred_thresh_enc,
            target_names=le.classes_,
            zero_division=0,
        )
    )
    res_thresh = {
        "name": "MLP + Grey Threshold",
        "accuracy": acc_thresh,
        "f1_macro": macro_f1_thresh,
        "y_pred": y_pred_thresh_enc,
    }
    results.append(res_thresh)

    # Comparison table
    print_summary_table(results)

    # Best model plots
    best = max(results, key=lambda x: x["f1_macro"])
    print(f"\nBest model: {best['name']}")

    # Confusion matrix for best model
    y_test_labels = le.inverse_transform(y_test_enc)
    y_pred_labels = le.inverse_transform(best["y_pred"])
    plot_confusion_matrix(
        y_test_labels,
        y_pred_labels,
        le.classes_.tolist(),
        PLOT_DIR / "confusion_matrix.png",
    )

    # Map name back to model object for feature importances
    model_map = {
        "k-NN (k=5)": knn,
        "Random Forest": rf,
        "SVM": svm,
        "HistGradientBoosting": hgb,
        "Logistic Regression": logreg,
        "MLP": mlp,
    }
    best_model = model_map.get(best["name"])
    if best_model is not None and hasattr(best_model, "feature_importances_"):
        plot_feature_importances(
            best_model.feature_importances_,
            FEATURE_COLS,
            PLOT_DIR / "feature_importances.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
