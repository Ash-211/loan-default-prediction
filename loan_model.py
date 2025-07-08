import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.shape)
print(train_df.columns)
print(train_df.head())
print(train_df.isnull().sum())

# Fill missing categorical values with mode
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(train_df[col].mode()[0])

# Fill missing numerical values
train_df['LoanAmount'] = train_df['LoanAmount'].fillna(train_df['LoanAmount'].median())
test_df['LoanAmount'] = test_df['LoanAmount'].fillna(train_df['LoanAmount'].median())

train_df['Loan_Amount_Term'] = train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mode()[0])
test_df['Loan_Amount_Term'] = test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0])

train_df['Credit_History'] = train_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0])
test_df['Credit_History'] = test_df['Credit_History'].fillna(train_df['Credit_History'].mode()[0])

print(train_df.isnull().sum())

# Feature Engineering
train_df['TotalIncome'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
test_df['TotalIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']

train_df['LoanAmount_log'] = np.log1p(train_df['LoanAmount'])
test_df['LoanAmount_log'] = np.log1p(test_df['LoanAmount'])

train_df['TotalIncome_log'] = np.log1p(train_df['TotalIncome'])
test_df['TotalIncome_log'] = np.log1p(test_df['TotalIncome'])

# Encode categorical features
cols_to_encode = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for col in cols_to_encode:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    print("Encoding:", le.classes_)

# Encode target variable
le = LabelEncoder()
train_df['Loan_Status'] = le.fit_transform(train_df['Loan_Status'])
print("Loan_Status Encoding:", le.classes_)

# Convert '3+' in Dependents column to int
train_df['Dependents'] = train_df['Dependents'].replace('3+', 3).astype(int)
test_df['Dependents'] = test_df['Dependents'].replace('3+', 3).astype(int)

# Drop unused columns
cols_to_drop = ['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome']
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)

# Split data
X = train_df.drop(columns=['Loan_Status'])
y = train_df['Loan_Status']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
print("Random Forest Accuracy:", accuracy_score(y_val, rf_preds))
print(classification_report(y_val, rf_preds))

# XGBoost - Learning rate tuning
best_lr = 0
best_acc = 0
lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3]

for lr in lrs:
    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=lr,
        early_stopping_rounds=10,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb.predict(X_val)
    acc = accuracy_score(y_val, xgb_preds)
    print(f"XGBoost LR: {lr}, Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_lr = lr
        best_acc = acc

# CatBoost - Learning rate tuning
for lr in lrs:
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=lr,
        depth=6,
        loss_function='Logloss',
        eval_metric='Accuracy',
        cat_features=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
        verbose=0,
        random_state=42
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    cat_preds = cat_model.predict(X_val)
    acc = accuracy_score(y_val, cat_preds)
    print(f"CatBoost LR: {lr}, Accuracy: {acc:.4f}")

# Final CatBoost Model (Best)
final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    cat_features=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],
    verbose=0,
    random_state=42
)
final_model.fit(X, y)
final_preds = final_model.predict(test_df)

# Prepare submission
submission = pd.DataFrame({
    'Loan_ID': pd.read_csv("test.csv")['Loan_ID'],
    'Loan_Status': le.inverse_transform(final_preds.astype(int))
})
submission.to_csv("submission.csv", index=False)
