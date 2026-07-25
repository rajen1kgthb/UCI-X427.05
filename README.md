# UCI Machine Learning Repository - Decision Tree Classification

## 📌 Project Overview
This repository contains the implementation and analysis of a **Decision Tree Classifier** using the **South German Credit** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). 

The objective of this assignment is to build a classification model to evaluate credit risk (`Good Credit` vs. `Bad Credit`), identify key predictive variables, visualize the resulting decision tree, and interpret the top decision splits.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Environment:** Google Colab
* **Libraries:**
  * `pandas` - Data manipulation & column mapping
  * `scikit-learn` - `DecisionTreeClassifier` model fitting & `plot_tree` visualization
  * `matplotlib` - Decision tree graphical rendering

---

## 📊 Dataset Information
* **Dataset:** [South German Credit](https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29)
* **Sample Size:** 1,000 instances
* **Target Variable:** `credit_risk` (`1` = Good Credit, `0` = Bad Credit)
* **Features:** 20 categorical and quantitative attributes (e.g., checking account status, loan duration, credit history, savings, employment duration).

---

## ⚙️ Model & Key Findings

### 1. Feature Importance
The trained decision tree identified the following top predictors driving credit risk classification:
1. **Checking Account Status (`status_checking_account`):** ~54.7% importance
2. **Duration in Months (`duration_months`):** ~14.6% importance
3. **Credit History (`credit_history`):** ~11.5% importance

### 2. Early Decision Rules
* **Root Split (Level 0):** Evaluates `status_checking_account <= 2.5`. Applicants with no checking account or higher balances are immediately routed to a low-risk classification path.
* **Level 1 Sub-Splits:**
  * For higher-risk checking accounts, the model splits on **loan duration** (`duration_months <= 22.5`). Shorter terms mitigate risk, while longer durations increase default likelihood.
  * For lower-risk checking accounts, the model evaluates **employment duration** (`employment_duration <= 3.5`) to confirm long-term stability.

---

## 🚀 How to Run
1. Open the `.ipynb` file in Google Colab directly via the badge link inside the notebook.
2. Run all code cells sequentially to reproduce the mapping, model training, feature importance table, and decision tree plot.
