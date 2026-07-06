from __future__ import annotations

import colorsys
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path("archive/legocolor-basic.csv")
COLORS_PATH = Path("archive/colors.csv")
ARTIFACTS_DIR = Path("artifacts")
RANDOM_STATE = 42
TEST_SIZE = 0.2
REQUIRED_COLUMNS = {"R", "G", "B", "Color", "Camera", "Photo_number", "X_axis", "Y_axis"}
BASE_FEATURES = [
    "R",
    "G",
    "B",
    "R_norm",
    "G_norm",
    "B_norm",
    "brightness",
    "max_channel",
    "min_channel",
    "channel_range",
    "hue",
    "saturation",
    "value",
]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, delimiter=";")
    pd.read_csv(COLORS_PATH, delimiter=",")

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {columns}")

    missing_values = data[list(REQUIRED_COLUMNS)].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(f"Dataset contains missing values:\n{missing_values.to_string()}")

    return data


def add_color_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()
    rgb = features[["R", "G", "B"]].astype(float)
    rgb_sum = rgb.sum(axis=1).replace(0, np.nan)

    features["R_norm"] = rgb["R"] / rgb_sum
    features["G_norm"] = rgb["G"] / rgb_sum
    features["B_norm"] = rgb["B"] / rgb_sum
    features["brightness"] = rgb.mean(axis=1)
    features["max_channel"] = rgb.max(axis=1)
    features["min_channel"] = rgb.min(axis=1)
    features["channel_range"] = features["max_channel"] - features["min_channel"]

    hsv_values = rgb.apply(
        lambda row: colorsys.rgb_to_hsv(row["R"] / 255, row["G"] / 255, row["B"] / 255),
        axis=1,
        result_type="expand",
    )
    hsv_values.columns = ["hue", "saturation", "value"]
    return pd.concat([features, hsv_values], axis=1).fillna(0)


def save_exploration_plots(data: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 6))
    order = data["Color"].value_counts().index
    sns.countplot(data=data, x="Color", order=order)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "class_distribution.png", dpi=160)
    plt.close()

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    labels = sorted(data["Color"].unique())
    palette = sns.color_palette("husl", len(labels))
    for label, color in zip(labels, palette, strict=True):
        subset = data[data["Color"] == label]
        axis.scatter(subset["R"], subset["G"], subset["B"], s=8, color=color, label=label)
    axis.set_xlabel("R")
    axis.set_ylabel("G")
    axis.set_zlabel("B")
    axis.legend(fontsize="x-small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "rgb_scatter_3d.png", dpi=160)
    plt.close()


def model_searches() -> dict[str, GridSearchCV]:
    numeric_preprocessor = ColumnTransformer(
        [("numeric", StandardScaler(), BASE_FEATURES)],
        remainder="drop",
    )
    mixed_preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), BASE_FEATURES),
            ("camera", OneHotEncoder(handle_unknown="ignore"), ["Camera"]),
        ],
        remainder="drop",
    )

    return {
        "dummy": GridSearchCV(
            Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
            param_grid={},
            scoring="f1_macro",
            cv=5,
        ),
        "knn": GridSearchCV(
            Pipeline([("preprocess", numeric_preprocessor), ("model", KNeighborsClassifier())]),
            param_grid={
                "model__n_neighbors": [3, 5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
            scoring="f1_macro",
            cv=5,
        ),
        "svc": GridSearchCV(
            Pipeline([("preprocess", numeric_preprocessor), ("model", SVC())]),
            param_grid={"model__C": [1, 10, 100], "model__gamma": ["scale", 0.1, 1]},
            scoring="f1_macro",
            cv=5,
        ),
        "logistic_regression": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", numeric_preprocessor),
                    ("model", LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)),
                ]
            ),
            param_grid={"model__C": [0.1, 1, 10]},
            scoring="f1_macro",
            cv=5,
        ),
        "random_forest": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", numeric_preprocessor),
                    ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            param_grid={
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 3],
            },
            scoring="f1_macro",
            cv=5,
        ),
        "extra_trees_with_camera": GridSearchCV(
            Pipeline(
                [
                    ("preprocess", mixed_preprocessor),
                    ("model", ExtraTreesClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            param_grid={
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 12, 24],
                "model__min_samples_leaf": [1, 2],
            },
            scoring="f1_macro",
            cv=5,
        ),
    }


def metrics_for(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(12, 10))
    display.plot(ax=axis, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def most_confused_pairs(
    y_true: pd.Series, y_pred: np.ndarray, labels: list[str]
) -> list[dict[str, int | str]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[dict[str, int | str]] = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if true_index != predicted_index and matrix[true_index, predicted_index] > 0:
                pairs.append(
                    {
                        "true": true_label,
                        "predicted": predicted_label,
                        "count": int(matrix[true_index, predicted_index]),
                    }
                )
    return sorted(pairs, key=lambda pair: int(pair["count"]), reverse=True)[:10]


def write_report(
    best_name: str,
    best_params: dict[str, object],
    results: dict[str, dict[str, object]],
    classification_summary: str,
    confused_pairs: list[dict[str, int | str]],
) -> None:
    lines = [
        "# LEGO Color Classification Report",
        "",
        "## Best model",
        "",
        f"- Model: `{best_name}`",
        f"- Parameters: `{best_params}`",
        "",
        "## Model comparison",
        "",
        "| Model | CV macro F1 | Test accuracy | Test precision macro | "
        "Test recall macro | Test macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name, result in results.items():
        metrics = result["test_metrics"]
        if not isinstance(metrics, dict):
            raise TypeError(f"Expected metrics for {model_name} to be a dictionary.")
        lines.append(
            "| "
            f"{model_name} | "
            f"{float(result['best_cv_f1_macro']):.4f} | "
            f"{float(metrics['accuracy']):.4f} | "
            f"{float(metrics['precision_macro']):.4f} | "
            f"{float(metrics['recall_macro']):.4f} | "
            f"{float(metrics['f1_macro']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Classification report",
            "",
            "```text",
            classification_summary.rstrip(),
            "```",
            "",
            "## Most confused color pairs",
            "",
        ]
    )

    for pair in confused_pairs:
        lines.append(f"- {pair['true']} → {pair['predicted']}: {pair['count']}")

    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
            "- `artifacts/class_distribution.png`",
            "- `artifacts/rgb_scatter_3d.png`",
            "- `artifacts/confusion_matrix.png`",
            "- `artifacts/metrics.json`",
            "- `artifacts/model.joblib`",
        ]
    )

    Path("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    print("[1/8] Loading datasets...")
    raw_data = load_data()
    print(f"      Loaded {len(raw_data)} samples from {DATA_PATH}.")

    print("[2/8] Engineering RGB, normalized RGB, brightness, and HSV features...")
    data = add_color_features(raw_data)

    print("[3/8] Saving exploration plots...")
    save_exploration_plots(data)
    print(f"      Plots saved to {ARTIFACTS_DIR}/.")

    print("[4/8] Inspecting class distribution...")
    print(data["Color"].value_counts().sort_index().to_string())

    x = data[BASE_FEATURES + ["Camera"]]
    y = data["Color"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        "[5/8] Creating stratified train/test split "
        f"({len(x_train)} train, {len(x_test)} test, random_state={RANDOM_STATE})..."
    )

    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_score = -1.0
    best_search: GridSearchCV | None = None
    searches = model_searches()

    print(f"[6/8] Training and tuning {len(searches)} models with 5-fold CV...")
    for index, (name, search) in enumerate(searches.items(), start=1):
        print(f"      ({index}/{len(searches)}) Fitting {name}...")
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)
        model_metrics = metrics_for(y_test, predictions)
        print(
            "          "
            f"CV macro F1={search.best_score_:.4f}; "
            f"test macro F1={model_metrics['f1_macro']:.4f}; "
            f"test accuracy={model_metrics['accuracy']:.4f}"
        )
        results[name] = {
            "best_cv_f1_macro": float(search.best_score_),
            "best_params": search.best_params_,
            "test_metrics": model_metrics,
        }
        if model_metrics["f1_macro"] > best_score:
            best_name = name
            best_score = model_metrics["f1_macro"]
            best_search = search

    if best_search is None:
        raise RuntimeError("No model was trained; cannot produce final evaluation.")

    print("[7/8] Evaluating best model and saving artifacts...")
    best_predictions = best_search.predict(x_test)
    labels = sorted(y.unique())
    report = classification_report(y_test, best_predictions, labels=labels, zero_division=0)
    pairs = most_confused_pairs(y_test, best_predictions, labels)

    save_confusion_matrix(y_test, best_predictions, labels)
    joblib.dump(best_search.best_estimator_, ARTIFACTS_DIR / "model.joblib")

    metrics_payload = {
        "best_model": best_name,
        "best_params": best_search.best_params_,
        "results": results,
        "most_confused_pairs": pairs,
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    write_report(best_name, best_search.best_params_, results, report, pairs)
    print("[8/8] Wrote REPORT.md and final metrics.")

    print(f"Best model: {best_name}")
    print(f"Best params: {best_search.best_params_}")
    print("\nTest metrics:")
    for metric_name, value in metrics_for(y_test, best_predictions).items():
        print(f"{metric_name}: {value:.4f}")
    print("\nClassification report:")
    print(report)
    print("Most confused pairs:")
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
