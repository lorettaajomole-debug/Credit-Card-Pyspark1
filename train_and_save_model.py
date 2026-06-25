import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import json

CSV = 'Credit_Card_Approval_10000_70_30.csv'
MODEL_OUT = 'pretrained_model.joblib'
META_OUT = 'pretrained_model_meta.json'

# Load
print('Loading CSV...')
df = pd.read_csv(CSV)
# Drop ids/dates if present
for col in ['application_id', 'application_date']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Prepare features
categorical_cols = [
    'employment_type', 'marital_status', 'education_level', 'residence_type', 'city_tier', 'card_type_requested'
]
numeric_cols = ['age', 'annual_income', 'credit_score', 'loan_amount', 'years_employed', 'debt_to_income_ratio', 'existing_loans']

for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())

# One-hot encode categoricals
existing_cats = [c for c in categorical_cols if c in df.columns]
if existing_cats:
    df_cat = pd.get_dummies(df[existing_cats].fillna('__MISSING__'))
else:
    df_cat = pd.DataFrame()

X = pd.concat([df[numeric_cols].fillna(0), df_cat], axis=1)
# target
y = (df['approval_status'].astype(str).str.lower().isin(['approved', 'accept', 'yes'])).astype(int)

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1] if hasattr(clf, 'predict_proba') else None
auc = roc_auc_score(y_test, probs) if probs is not None and len(set(y_test))>1 else None
acc = accuracy_score(y_test, preds)

meta = {'rows': len(X_test), 'auc': auc, 'accuracy': acc, 'feature_columns': list(X.columns)}

print('Saving model to', MODEL_OUT)
joblib.dump(clf, MODEL_OUT)
print('Saving metadata to', META_OUT)
with open(META_OUT, 'w') as f:
    json.dump(meta, f)

print('Done. Metrics:')
print(meta)
