<div align="center">

# 💻 Laptop Price Estimation: Linear vs. Regularized vs. Neural Regression

**A comparative Machine Learning study evaluating Baseline, L1/L2 Regularized Linear Models, and Multi-Layer Perceptrons (MLP) for hardware price prediction.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 📌 Project Overview

Predicting market prices of laptops is a regression task driven by both linear hardware scaling (e.g., RAM capacity, storage) and non-linear interactions (e.g., brand premiums, display PPI, discrete GPU tiers).

This project benchmarks a progression of regression architectures:
1. **Unregularized Baseline:** Standard Ordinary Least Squares (OLS) Linear Regression.
2. **L1 Regularization (Lasso):** Induces feature sparsity and automatic feature selection.
3. **L2 Regularization (Ridge):** Mitigates multicollinearity among hardware specs via weight decay.
4. **Non-Linear Deep Architecture (MLPRegressor):** A Multi-Layer Perceptron neural network optimized across multiple hidden layer configurations.

---

## 🛡️ Leakage-Free ML Methodology

To ensure strict scientific rigor and avoid **Data Leakage**, all transformations and scaling steps are bundled inside **`sklearn.pipeline.Pipeline`**:
- `StandardScaler` is fitted exclusively on training folds during Cross-Validation.
- Hyperparameter tuning is driven via **`GridSearchCV`** scoring on $R^2$.

```
 [ Train Folds ] ──► [ StandardScaler ] ──► [ Regressor ] ──► [ Cross-Validation Scoring ]
                                                                             │
 [ Held-out Test Split ] ───────────────────────────────────────────► [ Final evaluate_model() ]
```

---

## 🧠 Evaluated Models & Search Space

| Model | Pipeline Structure | Cross-Validation | Hyperparameter Grid / Search Space |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | `StandardScaler` + `LinearRegression()` | 5-Fold CV | Baseline (No hyperparameters) |
| **Lasso Regression (L1)** | `StandardScaler` + `Lasso(max_iter=50000, tol=0.01)` | 5-Fold CV | `alpha`: `[0.001, 0.01, 0.1, 1, 10, 100]` |
| **Ridge Regression (L2)** | `StandardScaler` + `Ridge(random_state=SEED)` | 5-Fold CV | `alpha`: `[0.1, 1, 10, 100, 200, 500, 1000]` |
| **MLP Perceptron (NN)** | `StandardScaler` + `MLPRegressor(max_iter=2000)` | 3-Fold CV | • `hidden_layer_sizes`: `[(64, 32), (100, 50), (128, 64, 32)]`<br>• `alpha`: `[0.0001, 0.001, 0.01, 0.1]`<br>• `learning_rate_init`: `[0.001, 0.01]` |

---

## 📊 Evaluation Metrics

All models are finally benchmarked on the unseen test set using a unified `evaluate_model` pipeline returning:
- **$R^2$ Score (Coefficient of Determination):** Explains the proportion of price variance captured.
- **MAE (Mean Absolute Error):** Measures average monetary prediction deviation in real currency units.
- **RMSE (Root Mean Squared Error):** Penalizes larger deviation errors caused by premium/luxury outliers.

---

## 📁 Repository Structure

```text
├── Laptop-Price.csv           # Raw dataset containing laptop hardware specs & target prices
├── PROGETTO_FINALE_ML.ipynb   # Complete Jupyter Notebook (EDA, Pipelines, GridSearchCV, Evaluation)
└── README.md                  # Project documentation
```

---

## 🚀 How to Run the Notebook

### 1. Clone the Repository
```bash
git clone https://github.com/BillyErMago/Laptop_comparison_ML.git
cd Laptop_comparison_ML
```

### 2. Set Up Environment & Dependencies
```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install jupyter pandas numpy scikit-learn matplotlib seaborn
```

### 3. Launch the Jupyter Notebook
```bash
jupyter notebook PROGETTO_FINALE_ML.ipynb
```

---

## 👤 Author

- **BillyErMago** - [GitHub Profile](https://github.com/BillyErMago)
