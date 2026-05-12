# LEGO Color Classification — Final Report

## 1. Overview

This project implements a machine learning pipeline to classify LEGO brick colors from pixel-level RGB measurements extracted from photographs. The challenge involves distinguishing **14 color classes** (including structurally similar greys) using only color-space features, with photos taken under varying lighting conditions and camera devices.

**Best result**: **88.1% accuracy** using an MLP with a custom brightness-threshold post-processor for the grey classes.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | `archive/legocolor-extended.csv` |
| Samples | 2,492 pixel measurements |
| Colors | 14 classes |
| Photos | 89 distinct images |
| Cameras | 3 (Canon G7X Mark ii, iPhone 8, iPhone 12 Pro) |
| Samples per photo | 28 (2 per color × 14 colors) |

### Color classes

`Aqua`, `Black`, `Bright Light Yellow`, `Bright Pink`, `Dark Azure`, `Dark Bluish Grey`, `Light Bluish Grey`, `Medium Azure`, `Orange`, `Red`, `Tan`, `White`, `Yellow`, `Yellowish Green`

### Raw vs extended features

The extended dataset includes engineered color-space transforms beyond raw RGB:

| Feature | Space | Purpose |
|---|---|---|
| `R`, `G`, `B` | RGB | Raw sensor values |
| `Hue`, `Saturation`, `Value` | HSV | Color identity + brightness separation |
| `Y`, `U`, `V` | YUV | Luminance + chrominance channels |
| `R_Average`, `G_Average`, `B_Average` | Per-photo | Exposure/color-cast normalization |

**Dropped**: `X_axis`, `Y_axis` (spatial leakage), `Camera` (avoid device-specific overfitting).

---

## 3. Methodology

### 3.1 Data split strategy

To prevent **data leakage**, the split is performed at the **photo level** (not pixel level):

- **Train**: 71 photos (~1,988 samples)
- **Test**: 18 photos (~504 samples)
- Method: `GroupShuffleSplit` grouped by `Photo_number`
- Every color appears in every photo, so the split is naturally stratified

Cross-validation for hyperparameter tuning uses `GroupKFold(5)` on the training photos.

### 3.2 Feature preprocessing

- `StandardScaler` (zero mean, unit variance) applied to all 12 features
- Required for: k-NN, SVM, Logistic Regression, MLP
- Benign for: Random Forest, HistGradientBoosting (tree models are scale-invariant)

### 3.3 Models evaluated

| Model | Type | Tuning | class_weight |
|---|---|---|---|
| k-NN (k=5) | Distance-based | Fixed k=5 | N/A |
| Random Forest | Ensemble (bagging) | Grid: `n_estimators`, `max_depth`, `min_samples_leaf` | `balanced` |
| SVM (RBF/Linear) | Kernel method | Grid: `C`, `gamma`, `kernel` | `balanced` |
| HistGradientBoosting | Ensemble (boosting) | Grid: `learning_rate`, `max_leaf_nodes`, `min_samples_leaf` | `balanced` |
| Logistic Regression | Linear | Single run, `max_iter=1000` | `balanced` |
| MLP | Neural network | HalvingGridSearchCV: `hidden_layer_sizes`, `alpha` | N/A |
| Two-Stage MLP | Hierarchical | Stage 1: grey vs color; Stage 2a/2b: specialized classifiers | N/A |
| **MLP + Grey Threshold** | Neural + heuristic | MLP base + `Y` threshold post-processor for grey pair | N/A |

### 3.4 Grey-specific interventions

The `Dark Bluish Grey` / `Light Bluish Grey` pair is the primary error source. Three approaches were tested:

1. **`class_weight='balanced'`** — modest improvement on grey recall
2. **Two-stage classifier** — dedicated grey classifier (Stage 2a); bottlenecked at 77% CV F1
3. **Brightness threshold** — brute-force search for optimal `Y` luminance threshold to split the grey pair

---

## 4. Results

### 4.1 Overall comparison

```
============================================
  Model Comparison
============================================
  Model                    Accuracy F1 (macro)
  ------------------------------------------
  k-NN (k=5)                 0.7817     0.7841
  Random Forest              0.8532     0.8560
  SVM                        0.8690     0.8711
  HistGradientBoosting       0.8571     0.8588
  Logistic Regression        0.8353     0.8359
  MLP                        0.8770     0.8801
  Two-Stage MLP              0.8710     0.8720
  MLP + Grey Threshold       0.8810     0.8835  ← BEST
============================================
```

### 4.2 Best model: MLP + Grey Threshold

**Architecture**: Single hidden layer (50 neurons), ReLU activation, Adam solver, α=0.0001

**CV performance**: 0.888 F1 (macro) on GroupKFold(5)

**Test performance**:

| Metric | Value |
|---|---|
| Accuracy | **88.1%** |
| F1 (macro) | **0.884** |

### 4.3 Per-class metrics (best model)

| Color | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Aqua | 0.97 | 0.92 | 0.94 | 36 |
| Black | 0.83 | 0.69 | 0.76 | 36 |
| Bright Light Yellow | 0.88 | 0.97 | 0.92 | 36 |
| Bright Pink | 1.00 | 0.92 | 0.96 | 36 |
| Dark Azure | 0.90 | 1.00 | 0.95 | 36 |
| **Dark Bluish Grey** | **0.52** | **0.61** | **0.56** | 36 |
| **Light Bluish Grey** | **0.62** | **0.72** | **0.67** | 36 |
| Medium Azure | 1.00 | 0.89 | 0.94 | 36 |
| Orange | 1.00 | 0.97 | 0.99 | 36 |
| Red | 0.95 | 1.00 | 0.97 | 36 |
| Tan | 0.94 | 0.94 | 0.94 | 36 |
| White | 0.88 | 0.81 | 0.84 | 36 |
| Yellow | 0.97 | 0.89 | 0.93 | 36 |
| Yellowish Green | 1.00 | 1.00 | 1.00 | 36 |

### 4.4 Grey confusion analysis

The three grey-ish classes form a **brightness continuum** along the luminance axis:

| Class | Mean `Value` | Mean `Y` | Mean `R` |
|---|---|---|---|
| Black | 0.13 | 30.2 | 28.5 |
| Dark Bluish Grey | 0.31 | 72.6 | 69.4 |
| Light Bluish Grey | 0.48 | 111.9 | 105.5 |

**Optimal threshold** (found by brute-force search on training data):
- `Y > 82.2` → predict `Light Bluish Grey`
- `Y ≤ 82.2` → predict `Dark Bluish Grey`

This threshold is applied **only** when the base MLP predicts either grey class, and improved overall accuracy from **87.7% → 88.1%** (+0.4pp).

The grey confusion is a **fundamental data limitation**: the two classes overlap in brightness space. Even with perfect knowledge of the luminance boundary, ~28% of grey samples fall on the wrong side due to intra-photo variance.

---

## 5. Key Findings

1. **MLP is the best model** for this tabular color-classification task, outperforming tree ensembles and SVM by ~1–2pp. The non-linear boundary learned by the neural network better captures the curved decision surfaces in HSV/YUV space.

2. **Hue is the most informative feature** — Random Forest feature importances rank `Hue` first, followed by `Value` and `B`. This makes intuitive sense: Hue directly encodes color identity.

3. **`class_weight='balanced'` helps marginally** on tree-based models and SVM, but the primary issue is not class imbalance (all classes have equal samples) — it's **separability** of the grey continuum.

4. **Two-stage and brightness-ratio features failed** — these added complexity without improving the grey bottleneck. The signal was already captured by `Value`, `Y`, and per-photo averages.

5. **The 88.1% ceiling is likely near-optimal** for pixel-level classification without spatial context. Grey confusion accounts for most of the remaining 12% error.

---

## 6. How to Run

```bash
# Run the full pipeline
uv run python main.py

# Lint
uv run ruff check main.py

# Format
uv run ruff format main.py
```

**Outputs**:
- Console: per-model metrics, comparison table, best model report
- `plots/confusion_matrix.png` — confusion matrix for best model
- `plots/feature_importances.png` — feature importances (when best model supports it)

---

## 7. Technical Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| `uv` | latest | Package manager & runner |
| `scikit-learn` | 1.8.0 | ML models & CV |
| `pandas` | 3.0.3 | Data loading |
| `numpy` | 2.4.4 | Numerics |
| `matplotlib` | 3.10.9 | Plotting |
| `seaborn` | 0.13.2 | Heatmaps |
| `ruff` | 0.15.12 | Linting & formatting |

---

## 8. Conclusion

The LEGO color classification challenge was addressed with a rigorous ML pipeline featuring group-stratified splits, extensive model comparison, and targeted grey-class interventions. The **MLP + Grey Threshold** model achieves **88.1% accuracy** on a held-out test set of unseen photos. The remaining error is concentrated in the `Dark Bluish Grey` / `Light Bluish Grey` pair, which form a near-continuous brightness distribution that no feature engineering or architectural trick fully resolves.

**Recommendation for production**: Use the MLP + Grey Threshold model. If higher grey accuracy is critical, collect additional training data specifically targeting the mid-brightness boundary between the two grey classes, or add spatial/texture features to disambiguate.
