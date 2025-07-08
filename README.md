# Loan Default Prediction Model

This project predicts whether a loan will be approved based on applicant details using various machine learning models. The dataset is available here:  
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

## Folder Structure

Loan-Default-Prediction/
├── loan_model.py
├── submission.csv
├── train.csv
├── test.csv
├── requirements.txt
└── .gitignore
## Dataset

- `train.csv` — training data with `Loan_Status` labels  
- `test.csv` — test data without labels (used for submission)

## Features Used

Categorical:
- Gender
- Married
- Dependents
- Education
- Self_Employed
- Property_Area

Numerical:
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History

Engineered:
- TotalIncome
- LoanAmount_log
- TotalIncome_log

## Preprocessing Steps

- Missing values filled with median or mode
- Feature engineering: `TotalIncome`, `LoanAmount_log`, `TotalIncome_log`
- Label encoding for categorical variables
- Converted `3+` in `Dependents` to `3`

## Models Trained

- Random Forest Classifier
- XGBoost Classifier (with learning rate tuning)
- CatBoost Classifier (with learning rate tuning)

The final model used for submission is CatBoostClassifier with `learning_rate=0.1`.

## Results

| Model        | Accuracy (Validation) |
|--------------|-----------------------|
| RandomForest | ~0.82                 |
| XGBoost      | ~0.84                 |
| CatBoost     | ~0.87 (Best)          |

## Output

The model produces a `submission.csv` file which contains predictions (`Loan_ID`, `Loan_Status`) for the test data.

## How to Run

1. Install dependencies:
        _pip install -r requirements.txt_

2. Ensure the following files are in the same directory:
- `train.csv`
- `test.csv`
- `loan_model.py`

3. Run the script:
        *python load_model.py*
