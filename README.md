# 🛠️ ML Data Cleaning Skills

**Reusable Python toolkit for data cleaning, preprocessing, and machine learning model training — built from practical experience and lessons learned.**

[![Python](https://img.shields.io/badge/Language-Python_3-3776AB?logo=python&logoColor=white)]()
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-FF6F00?logo=tensorflow&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas&logoColor=white)]()

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Function Reference](#-function-reference)
- [Key Concepts Learned](#-key-concepts-learned)
- [Usage Example](#-usage-example)
- [Dependencies](#-dependencies)
- [Resources](#-resources)

---

## 🔬 Overview

This repository contains my **personal reusable functions** for data cleaning, preprocessing, and ML model training. These functions were developed and refined through practical work, and are designed to be **modular**, **importable**, and **reusable** across different projects.

The toolkit covers the full ML pipeline:

```
Raw Data → Cleaning → Preprocessing → Model Training → Evaluation
```

---

## 📂 Project Structure

```
ml-data-cleaning-skills/
│
├── my_functions/                               # 🌟 Personal reusable functions
│   ├── data_cleaning.py                        # Data cleaning & exploration
│   ├── preprocessing.py                        # Standardization, encoding, splits
│   └── ml_models.py                            # ML model training & evaluation
│
├── notebooks/                                  # Reference notebooks
│   ├── readmefile                              # Notebook documentation
│   └── PracticalWork_DataCleaning_clean.ipynb  # Original practical work (reference)
│
└── README.md                                   # Project documentation
```

---

## 📋 Function Reference

### 🧹 `data_cleaning.py` — Data Cleaning & Exploration

| Function | Purpose |
|:---|:---|
| `load_and_explore(filepath)` | Load a CSV and display head, info, describe, nunique |
| `drop_irrelevant_columns(df, cols)` | Remove unnecessary columns |
| `handle_missing_values(df, strategy)` | Handle NaN: `'drop'`, `'fill'`, or `'mean'` |
| `bin_column(df, col, bins, labels)` | Transform continuous column into categories |
| `fix_category_typo(df, col, wrong, correct)` | Fix typos in categorical values |
| `add_ratio_features(df)` | Create ratio-based features (feature engineering) |
| `drop_redundant_columns(df, threshold)` | Remove columns with high inter-correlation |

---

### ⚙️ `preprocessing.py` — ML Preparation

| Function | Purpose |
|:---|:---|
| `simple_split(df, test_size)` | Random train/test split |
| `stratified_split(df, col, test_size)` | Stratified split (preserves class proportions) |
| `standardize_numerical(df, scaler)` | Standardize numerical features (mean=0, std=1) |
| `one_hot_encode(df, categorical_cols)` | One-hot encode categorical variables |
| `process_data(dataset, target, cat_cols)` | Full pipeline: X/Y split + standardization + one-hot |
| `to_tensorflow_dataset(X, Y, n_classes)` | Convert to TensorFlow dataset for `model.fit()` |

---

### 🤖 `ml_models.py` — Machine Learning Models

| Function | Type | Purpose |
|:---|:---|:---|
| `train_linear_regression(...)` | Regression | Linear baseline, computes RMSE |
| `train_decision_tree_regressor(...)` | Regression | Decision tree + feature importances |
| `train_svr(...)` | Regression | Support Vector Regression |
| `train_decision_tree_classifier(...)` | Classification | Decision tree + classification report |
| `train_random_forest(...)` | Classification | Ensemble of multiple trees |
| `train_mlp_sklearn(...)` | Classification | Neural network (scikit-learn) |
| `build_tensorflow_model(...)` | Classification | Neural network (TensorFlow/Keras) |
| `train_tensorflow_model(...)` | Classification | Train with early stopping |
| `plot_training_history(history)` | Visualization | Train vs validation loss curves |

---

## 📖 Key Concepts Learned

### 🔴 Data Leakage — The Most Important Pitfall

Data leakage occurs when information from the test set "leaks" into training. The model appears to perform well during evaluation but fails in production.

```python
# ❌ WRONG: scaler sees test data → data leakage!
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)    # Never re-fit on test!

# ✅ CORRECT: fit only on train, transform both
scaler.fit_transform(X_train)   # learns mean and std from train
scaler.transform(X_test)        # applies without re-learning
```

---

### 🔴 One-Hot Encoding vs Label Encoding

```python
# ❌ Label Encoding for nominal categories
# "Aerial"=1, "Aquatic"=2, "Terrestrial"=3
# → model thinks Terrestrial(3) > Aquatic(2) > Aerial(1)
# This is NOT what we want!

# ✅ One-Hot Encoding
# "Aerial"      → [1, 0, 0]
# "Aquatic"     → [0, 1, 0]
# "Terrestrial" → [0, 0, 1]
# No ordinal relationship → correct for nominal categories
```

---

### 🔴 Stratified vs Random Split

```python
# ❌ Random split with rare classes
# If "Scavenger" = 0.3% of data and dataset = 10,000 rows
# → only ~30 Scavengers. Test set may have NONE!

# ✅ Stratified split
# Guarantees each class keeps its proportion in train AND test
from sklearn.model_selection import StratifiedShuffleSplit
```

---

### 🔴 Overfitting — Detecting It with Loss Curves

```
Loss
│  \
│   \  ← train loss decreasing  ✅
│    \_______
│            \____
│                 \   ← val loss rising  ❌ = OVERFITTING
│
└────────────────── Epochs
```

**Solution:** Early Stopping — stop training when `val_loss` no longer improves.

---

### 🔴 Standardization vs Normalization

| | Standardization (`StandardScaler`) | Normalization (`MinMaxScaler`) |
|:---|:---|:---|
| **Result** | mean=0, std=1 | values in [0, 1] |
| **Advantage** | Preserves distribution shape | Easy to interpret |
| **Disadvantage** | Less intuitive | Very sensitive to outliers |
| **When to use** | Statistical models, SVM, MLP | When outliers are controlled |

---

## 🚀 Usage Example

```python
# Import modules
from my_functions.data_cleaning import load_and_explore, drop_irrelevant_columns, handle_missing_values
from my_functions.preprocessing import stratified_split, process_data
from my_functions.ml_models import train_random_forest, build_tensorflow_model

# 1. Load and explore
df = load_and_explore('AVONET.csv')

# 2. Clean
df = drop_irrelevant_columns(df, ['Family3', 'Order3', 'Total.individuals'])
df = handle_missing_values(df, strategy='drop')

# 3. Split (stratified)
train_set, test_set = stratified_split(df, stratify_col='Trophic.Niche', test_size=0.25)

# 4. Prepare features
cat_cols = ['Habitat', 'Primary.Lifestyle', 'Centroid.Longitude']
X_train, Y_train, scaler = process_data(train_set, 'Trophic.Niche', cat_cols)
X_test,  Y_test,  _      = process_data(test_set,  'Trophic.Niche', cat_cols, scaler=scaler)

# 5. Train a model
model, preds, report = train_random_forest(X_train, Y_train, X_test, Y_test)
```

---

## 📦 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

| Library | Purpose |
|:---|:---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | ML models, preprocessing, metrics |
| `tensorflow` | Neural network training |

---

## 🔗 Resources

- [Scikit-Learn — Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-Learn — Model Selection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)
- [TensorFlow — Keras Sequential Model](https://www.tensorflow.org/guide/keras/sequential_model)
- [Towards Data Science — Data Leakage](https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742)
