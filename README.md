<div align="center">

# 💻 Laptop Price Prediction & ML Benchmarking

**An end-to-end Machine Learning regression study predicting market prices of laptops through feature engineering and comparative model evaluation.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

</div>

---

## 📌 Project Overview

Hardware pricing is heavily driven by complex non-linear interactions between components (CPU clock speeds, RAM size/type, storage combinations, GPU tier, display resolution, and brand equity).

This project formulates **Laptop Price Estimation** as a supervised regression task. Using an empirical dataset of laptops (`Laptop-Price.csv`), it implements a complete data science pipeline:
1. **Exploratory Data Analysis (EDA):** Identifying price distributions, component correlations, and outliers.
2. **Feature Engineering:** Parsing raw strings (e.g., CPU manufacturer/clock speed, SSD/HDD storage splits, resolution PPI) into structured numerical features.
3. **Model Benchmarking:** Training and comparing multiple linear and ensemble regression models to find the highest predictive accuracy.

---

## 🔄 Machine Learning Pipeline

```
  [ Raw Dataset (CSV) ]
            │
            ▼
 1. Data Cleaning & Parsing   ───► Extracts RAM in GB, separates SSD/HDD, parses CPU/GPU
            │
            ▼
 2. Exploratory Data Analysis ───► Correlation heatmaps, feature distributions, outlier handling
            │
            ▼
 3. Feature Transformation    ───► One-Hot Encoding (Brands/OS), Log-transform (Target Price), Scaling
            │
            ▼
 4. Model Training & Tuning   ───► Evaluates multiple regressors via Cross-Validation & GridSearch
            │
            ▼
 5. Evaluation & Comparison   ───► Compares R² Score, MAE, and RMSE across models
```

---

## 🧠 Evaluated Models

The benchmark compares traditional linear baselines against non-linear tree-based ensembles:

- **Linear Models:** Linear Regression, Ridge, Lasso
- **Instance-Based:** K-Nearest Neighbors (KNN) Regressor
- **Support Vector Machines:** SVR (Support Vector Regressor)
- **Tree Ensembles:** Decision Trees, Random Forest Regressor, Gradient Boosting / XGBoost

### Evaluation Metrics
Models are evaluated on held-out test splits using:
- **$R^2$ Score (Coefficient of Determination):** Proportion of variance explained by the model.
- **MAE (Mean Absolute Error):** Average magnitude of prediction error in monetary terms.
- **RMSE (Root Mean Squared Error):** Penalty metric prioritizing larger deviation errors.

---

## 📁 Repository Structure

```text
├── Laptop-Price.csv           # Raw dataset containing laptop hardware specs & prices
├── PROGETTO_FINALE_ML.ipynb   # Complete Jupyter Notebook (EDA, preprocessing, training, evaluation)
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started & How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/BillyErMago/Laptop_comparison_ML.git
cd Laptop_comparison_ML
```

### 2. Set Up Environment & Install Dependencies
Create a virtual environment and install the required scientific stack:

```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install jupyter pandas numpy scikit-learn matplotlib seaborn
```

### 3. Launch the Notebook
Start Jupyter Notebook or JupyterLab to inspect the interactive analysis and training steps:

```bash
jupyter notebook PROGETTO_FINALE_ML.ipynb
```

---

## 👤 Author

- **BillyErMago** - [GitHub Profile](https://github.com/BillyErMago)
