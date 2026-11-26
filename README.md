# ChurnIQ – Customer Churn Prediction System

A Streamlit machine-learning project using the IBM Telco Customer Churn dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data preprocessing file

The project includes `data_preprocessing.py`, a standalone, commented preprocessing program based on the preprocessing work in `IA2(1).ipynb`.

Run it from this project folder:

```bash
python data_preprocessing.py
```

It creates `data/processed/` with:
- `telco_churn_cleaned.csv`
- encoded train/test feature files for churn classification
- train/test target files for churn classification
- encoded train/test feature files for monthly-charge regression
- train/test target files for monthly-charge regression

Preprocessing includes whitespace cleanup, conversion of `TotalCharges` to numeric, median imputation, duplicate removal, feature selection, train/test splitting, one-hot encoding and numerical standardization. The encoders/scalers are fitted only on training data to reduce data leakage.

## Application pages

1. Dashboard – dataset KPIs, four visualisations and business insights.
2. Data Explorer – filters, segment analysis and CSV export.
3. Data Preprocessing – raw data quality, preprocessing reasons, cleaned preview and descriptive statistics.
4. Predict Churn – churn prediction, risk score, retention actions and similar customers.
5. Predict Charges – monthly charge regression.
6. Model Performance – classification comparison, ROC curves and confusion matrices.
7. K-Fold Validation – Stratified K-Fold results for classification and K-Fold for regression.
8. KNN Lab – K tuning and nearest-neighbour inspection.
9. Reports & Export – metrics exports and examination rubric/viva coverage.

## Practical examination rubric mapping (A–I)

| Requirement | Project evidence |
|---|---|
| A. Data Collection | `data/telco_churn.csv`, Data Explorer dataset overview |
| B. Descriptive Statistics | Data Preprocessing page and `df.describe()` output |
| C. Exploration & Visualisation | Dashboard, Data Explorer and Streamlit BI-style dashboard |
| D. Data Preparation | Data Preprocessing page and `data_preprocessing.py` |
| E. Splitting & Cross-Validation | Hold-out split, Stratified K-Fold and regression K-Fold |
| F. Model Implementation & Tuning | Decision Tree, Random Forest, Logistic Regression, KNN and Linear Regression; configurable KNN/tree/forest settings |
| G. Model Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, MAE, RMSE and R²; model comparison |
| H. Deployment | Streamlit deployment-ready app; see deployment notes below |
| I. Presentation & Viva | Rubric/viva notes in Reports page; use app for live demonstration |

## Deployment (Streamlit Community Cloud)

1. Push the complete project folder to a GitHub repository.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select the repository, branch and `app.py`.
4. Deploy. Keep `requirements.txt` in the repository root.
5. Test the deployed application with the bundled data and any permitted uploaded CSV.

For production use, protect customer data, restrict access, avoid exposing personal information, monitor data quality and model drift, and explain that predictions are probabilistic rather than certain.

## Important examination submission reminders

The practical PDF requires source code with comments, a PDF report containing student details/problem statement/GitHub link/code/output screenshots/explanations, presentation slides, a poster, an article and the GitHub repository link. These are separate submission artifacts; the Streamlit application does not automatically create all of them. Add your own name, seat number, PRN and repository link before submission.

## Dataset note

The bundled dataset has 7,043 customer records and 21 original attributes. The project has two tasks:
- Classification: predict `Churn` (Yes/No).
- Regression: predict `MonthlyCharges`.

The regression feature selection excludes `TotalCharges` because it is closely related to the monthly billing target and could cause target-related leakage. The churn classifier retains billing features as predictors.
