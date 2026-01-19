# Heart Disease Prediction using Machine Learning
 
## Project Overview
Heart disease is one of the leading causes of death worldwide. This project uses **machine learning** to predict the likelihood of heart disease based on **clinical & demographic patient data**.

The system includes:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Multiple machine learning models
- Web deployment using **Flask** & **Streamlit**

---

## Dataset
- **Records:** 270 patients
- **Features:** 13 clinical attributes
- **Target:** Heart Disease (Presence / Absence)

### Target Distribution
- Absence: **55.5%**
- Presence: **44.5%**

---

## Exploratory data analysis (EDA)

Key findings:
- Asymptomatic chest pain strongly correlates with heart disease
- Higher ST depression and number of affected vessels increase risk
- Max heart rate shows an inverse relationship with disease presence
- Males show higher disease prevalence than females

---

## Feature engineering
- Target encoding: Presence -> "1", Absence -> "0"
- Standardization using "StandardScaler"
- No missing values in dataset
- Train-test split: **80% / 20%**

---

## Models Implemented

### Logistic Regression
- Accuracy: **85.18%**
- AUC: **0.898**
- Strong recall for disease detection

### Random Forest Classifier
- Accuracy: **83.33%**
- AUC: **0.898**
- Robust non-linear performance

---

## Model Evaluation

### Logistic Regression - Classification Report

|   Class   | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| Absence   |   0.92    |  0.80  |   0.86   |
| Presence  |   0.79    |  0.92  |   0.85   |

### Random Forest - Classification Report

|   Class   | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| Absence   |   0.86    |  0.83  |   0.85   |
| Presence  |   0.80    |  0.83  |   0.82   |

---

## Feature Importance

### Random Forest (Top Predictors)
- Chest pain type
- Thallium scan
- Max heart rate
- Number of vessels
- ST depression

### Logistic Regression (Key Coefficients)
- Number of vessels (positive impact)
- Chest pain type
- Exercise-induced angina
- Max heart rate (negative impact)

---

## Web Application

### Flask Web App
- Medical-themed UI
- Patient data input form
- Probability-based prediction output

### Streamlit App
- Interactive interface
- Example patient selection
- Real-time prediction

---

## How to Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/AshitoshR/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirenments.txt
```

### 3. Run Flask app
```bash
cd flask_app
python app.py
```

### 4. Run streamlit app
```bash
cd streamlit_app
streamlit run app.py
```

---

## Technologies Used
- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn
- Flask
- Streamlit

## Furture Improvements
- SHAP-based explainability
- Deep Learning models
- Docker deployment
- Cloud hosting

---

## Author
Ashitosh Rokade