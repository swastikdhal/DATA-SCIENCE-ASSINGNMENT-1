import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10

print("=" * 60)
print("1. LOADING RAW DATASETS")
print("=" * 60)
raw_train = pd.read_csv("train.csv")
raw_test = pd.read_csv("test.csv")

print(f"Train Raw Shape: {raw_train.shape[0]:,} rows, {raw_train.shape[1]} columns")
print(f"Test Raw Shape:  {raw_test.shape[0]:,} rows, {raw_test.shape[1]} columns\n")

missing_counts = raw_train.isnull().sum()
missing_summary = pd.DataFrame({
    'Missing Records': missing_counts[missing_counts > 0],
    'Percentage (%)': (missing_counts[missing_counts > 0] / len(raw_train)) * 100
})
print("--- Missing Values Audit (Raw Train) ---")
print(missing_summary.to_string())
print()

def clean_duration_to_months(series: pd.Series) -> pd.Series:
    years = series.str.extract(r'(\d+)yrs')[0].astype(float).fillna(0)
    months = series.str.extract(r'(\d+)mon')[0].astype(float).fillna(0)
    return (years * 12 + months).astype(int)

def clean_dataset(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    
    df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'], format='%d-%m-%y')
    df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'], format='%d-%m-%y')
    df['Date.of.Birth'] = df['Date.of.Birth'].apply(
        lambda d: d.replace(year=d.year - 100) if d.year > 2018 else d
    )
    
    df['age_at_disbursal'] = ((df['DisbursalDate'] - df['Date.of.Birth']).dt.days // 365.25).astype(int)
    df['avg_acct_age_months'] = clean_duration_to_months(df['AVERAGE.ACCT.AGE'])
    df['credit_history_months'] = clean_duration_to_months(df['CREDIT.HISTORY.LENGTH'])
    df['no_bureau_history'] = (df['PERFORM_CNS.SCORE'] == 0).astype(int)
    df['delinquency_ratio'] = df['PRI.OVERDUE.ACCTS'] / (df['PRI.ACTIVE.ACCTS'] + 1)
    
    df['Employment.Type'] = df['Employment.Type'].fillna('Missing')
    
    drop_cols = [
        'MobileNo_Avl_Flag', 'Date.of.Birth', 'DisbursalDate',
        'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH'
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])

print("2. EXECUTING DATA CLEANING & FEATURE ENGINEERING...")
cleaned_train = clean_dataset(raw_train)
cleaned_test = clean_dataset(raw_test)

audit_table = pd.DataFrame({
    'Metric': ['Total Rows', 'Total Columns', 'Total Missing Cells', 'Duplicate Rows'],
    'Raw Train': [f"{len(raw_train):,}", len(raw_train.columns), f"{raw_train.isnull().sum().sum():,}", f"{raw_train.duplicated().sum():,}"],
    'Cleaned Train': [f"{len(cleaned_train):,}", len(cleaned_train.columns), f"{cleaned_train.isnull().sum().sum():,}", f"{cleaned_train.duplicated().sum():,}"]
})
print("\n--- Cleaning Audit Report ---")
print(audit_table.to_string(index=False))
print()

print("3. GENERATING & DISPLAYING INDIVIDUAL EDA PLOTS...")

plt.figure(figsize=(7, 5))
sns.countplot(
    x='loan_default',
    hue='loan_default',
    data=cleaned_train,
    palette='Blues_r',
    legend=False
)
plt.title('Loan Default Distribution (0 = No Default, 1 = Default)', fontweight='bold')
plt.xlabel('Default Flag')
plt.ylabel('Applicant Count')
plt.tight_layout()
plt.savefig("eda_1_target_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ Saved & displayed 'eda_1_target_distribution.png'")

plt.figure(figsize=(7, 5))
sns.histplot(cleaned_train['disbursed_amount'], bins=40, kde=True, color='#1f77b4')
plt.title('Distribution of Disbursed Loan Amount', fontweight='bold')
plt.xlabel('Disbursed Amount (₹)')
plt.xlim(10000, 200000)
plt.tight_layout()
plt.savefig("eda_2_disbursed_amount.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ Saved & displayed 'eda_2_disbursed_amount.png'")

plt.figure(figsize=(7, 5))
sns.boxplot(
    x='loan_default',
    hue='loan_default',
    y='age_at_disbursal',
    data=cleaned_train,
    palette='Set2',
    legend=False
)
plt.title('Customer Age by Default Status', fontweight='bold')
plt.xlabel('Default Flag')
plt.ylabel('Age at Disbursal (Years)')
plt.tight_layout()
plt.savefig("eda_3_age_vs_default.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ Saved & displayed 'eda_3_age_vs_default.png'")

plt.figure(figsize=(7, 5))
sns.histplot(cleaned_train['PERFORM_CNS.SCORE'], bins=40, kde=False, color='#2ca02c')
plt.title('Bureau Credit Score Distribution (0 = No History)', fontweight='bold')
plt.xlabel('Bureau Score (300 - 900)')
plt.tight_layout()
plt.savefig("eda_4_bureau_score.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print("✓ Saved & displayed 'eda_4_bureau_score.png'")

print("\n" + "=" * 50)
print("          VEHICLE LOAN PORTFOLIO SUMMARY          ")
print("=" * 50)
print(f"Total Applicants Evaluated : {len(cleaned_train):,}")
print(f"Overall Default Rate       : {cleaned_train['loan_default'].mean() * 100:.2f}%")
print(f"Average Disbursed Amount   : ₹{cleaned_train['disbursed_amount'].mean():,.2f}")
print(f"Average Applicant Age      : {cleaned_train['age_at_disbursal'].mean():.1f} years")
print(f"Applicants Without Bureau  : {cleaned_train['no_bureau_history'].mean() * 100:.2f}%")
print("=" * 50 + "\n")

print("4. PREPARING FEATURES & TRAINING MODEL...")
id_cols_to_exclude = ['UniqueID', 'supplier_id', 'Current_pincode_ID', 'Employee_code_ID', 'branch_id']
X = cleaned_train.drop(columns=id_cols_to_exclude + ['loan_default'])
y = cleaned_train['loan_default']

cat_features = ['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION', 'manufacturer_id', 'State_ID']
for col in cat_features:
    X[col] = X[col].astype(str)

num_features = [col for col in X.columns if col not in cat_features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
])

model = Pipeline([
    ('prep', preprocessor),
    ('clf', HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42))
])

model.fit(X_train, y_train)

val_probs = model.predict_proba(X_val)[:, 1]
val_preds = model.predict(X_val)
auc_val = roc_auc_score(y_val, val_probs)

print(f"\n✓ Validation ROC-AUC Score: {auc_val:.4f}")
print("\n--- Validation Classification Report ---")
print(classification_report(y_val, val_preds, digits=4))

plt.figure(figsize=(7, 5))
fpr, tpr, _ = roc_curve(y_val, val_probs)
plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'HistGradientBoosting (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved 'roc_curve.png' to project folder.")

print("\n5. GENERATING TEST PREDICTIONS...")
X_test = cleaned_test.drop(columns=id_cols_to_exclude)
for col in cat_features:
    X_test[col] = X_test[col].astype(str)

test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.50).astype(int)

submission = pd.DataFrame({
    'UniqueID': raw_test['UniqueID'],
    'loan_default_probability': np.round(test_probs, 4),
    'loan_default_prediction': test_preds
})

submission.to_csv("submission_predictions.csv", index=False)
print("✓ Successfully created 'submission_predictions.csv' (112,392 predictions).")
print("\nSample Predictions:")
print(submission.head())