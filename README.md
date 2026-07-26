# 🌳 Decision Tree Classification — End-to-End Machine Learning with the Adult Income Dataset

An end-to-end **Machine Learning Classification Project** using the **Adult Income Census Dataset**. The objective is to predict whether an individual's annual income exceeds **$50,000** based on demographic and employment-related features using a **Decision Tree Classifier**.

---

## 📌 Project Overview

This project demonstrates the complete workflow of a supervised machine learning project:

- Data Loading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Train/Test Split
- Decision Tree Modeling
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance Analysis
- Model Persistence using Joblib

---

## 🎯 Problem Statement

Given demographic and employment information from the U.S. Census Adult Income dataset, predict whether an individual earns:

- **>50K**
- **<=50K**

This is a **binary classification problem**.

---

## 📂 Dataset

Dataset: **Adult Income Dataset (UCI Census Income Dataset)**

The dataset contains attributes such as:

- Age
- Workclass
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country
- Income Class (Target)

Target Variable:

```
class
```

Converted into

```
1 → Income >50K
0 → Income <=50K
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Joblib

---

# 📁 Project Structure

```
Decision-Tree-Adult-Income/
│
├── phpMawTba.csv
├── decision_tree_income.joblib
├── DecisionTree_AdultIncome.ipynb
├── README.md
└── requirements.txt
```

---

# 📊 Exploratory Data Analysis

The notebook performs extensive EDA including:

### ✔ Dataset Information

- Shape
- Data Types
- Missing Values
- Summary Statistics

### ✔ Class Distribution

Count plot showing class imbalance.

### ✔ Numerical Feature Analysis

Histogram + KDE plots

- Age
- fnlwgt
- education-num
- capital-gain
- capital-loss
- hours-per-week

### ✔ Boxplots

Comparison of feature distributions across income classes.

### ✔ Categorical Analysis

Income rate by

- Education
- Occupation
- Marital Status
- Sex

### ✔ Correlation Heatmap

Correlation among numerical variables and target.

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Missing values removed
- Target converted into binary labels
- One-Hot Encoding using `pd.get_dummies()`
- Boolean columns converted to integers
- Train/Test split using Stratified Sampling

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)
```

---

# 🌳 Decision Tree Models

Three different decision trees were trained.

## 1. Shallow Tree

```
max_depth = 3
```

Purpose:

- Demonstrates **Underfitting**

---

## 2. Fully Grown Tree

```
DecisionTreeClassifier()
```

Purpose:

- Demonstrates **Overfitting**

---

## 3. Tuned Decision Tree

Optimized using hyperparameter tuning.

Example:

```
max_depth = 9
min_samples_leaf = 20
```

This model provides the best trade-off between bias and variance.

---

# 🔍 Hyperparameter Tuning

Several tuning strategies were explored.

## Brute Force Search

Testing

```
max_depth = 1 → 25
```

Visualized with train/test accuracy curves.

---

## Validation Curve

5-Fold Cross Validation using ROC-AUC.

```
validation_curve()
```

---

## GridSearchCV

Parameters searched:

- max_depth
- min_samples_leaf

Scoring metric:

```
ROC-AUC
```

---

## RandomizedSearchCV

Random search over:

- criterion
- max_depth
- min_samples_leaf
- min_samples_split
- max_features

---

# 📈 Model Evaluation

Metrics computed:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Example:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
roc_auc_score()
```

---

# 📉 Confusion Matrix

The project visualizes a confusion matrix showing:

- True Positives
- True Negatives
- False Positives
- False Negatives

using Seaborn heatmaps.

---

# ⭐ Feature Importance

The trained Decision Tree provides feature importance values.

Top important features are displayed and ranked, helping explain the model's decisions.

---

# 💾 Model Persistence

The trained model is saved using Joblib.

```python
joblib.dump(tree, "decision_tree_income.joblib")
```

Reloading:

```python
tree_loaded = joblib.load("decision_tree_income.joblib")
```

The notebook verifies that the loaded model produces identical predictions.

---

# 📚 Concepts Covered

- Classification
- Decision Trees
- Entropy
- Gini Index
- Underfitting
- Overfitting
- Train/Test Split
- Cross Validation
- Grid Search
- Random Search
- ROC-AUC
- Feature Importance
- Model Serialization

---

# 🚀 How to Run

### Clone the repository

```bash
git clone https://github.com/yourusername/Decision-Tree-Adult-Income.git
```

### Navigate to the project

```bash
cd Decision-Tree-Adult-Income
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
DecisionTree_AdultIncome.ipynb
```

Run all cells sequentially.

---

# 📦 Required Libraries

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
joblib
```

Install them using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
```

---

# 📊 Workflow

```
Dataset
    │
    ▼
Load Data
    │
    ▼
EDA
    │
    ▼
Data Cleaning
    │
    ▼
One-Hot Encoding
    │
    ▼
Train/Test Split
    │
    ▼
Decision Tree Models
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Model Evaluation
    │
    ▼
Feature Importance
    │
    ▼
Save Model
```

---

# 🎯 Key Takeaways

- Learned the complete machine learning workflow for a classification problem.
- Compared shallow, fully grown, and tuned decision trees.
- Demonstrated underfitting and overfitting.
- Improved model performance through hyperparameter tuning.
- Evaluated the model using multiple performance metrics.
- Saved and reloaded the trained model for future predictions.

---

## 👨‍💻 Author

**Ridima**

GitHub: https://github.com/Ridima28

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

⭐ If you found this project useful, consider giving the repository a **star** on GitHub!

