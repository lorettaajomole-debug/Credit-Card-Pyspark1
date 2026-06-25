"""
A small training script that prefers PySpark but falls back to pandas + scikit-learn
when PySpark is not available (useful for environments like Streamlit Cloud).

Usage:
  - Local with PySpark: install pyspark in your virtualenv and run the script.
  - Streamlit / lightweight: install requirements.txt (pandas, scikit-learn, streamlit)
    and the script will use scikit-learn instead.
"""

import sys
from pathlib import Path

DATA_PATH = "Credit_Card_Approval_10000_70_30.csv"

USE_PYSPARK = True
try:
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import RandomForestClassifier as SparkRF
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
    from pyspark.sql import SparkSession
except Exception:
    USE_PYSPARK = False

if not USE_PYSPARK:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score


def run_pyspark(path: str):
    spark = SparkSession.builder.master("local[*]").appName("CreditCardApprovalClassification").getOrCreate()
    raw_df = spark.read.csv(path, header=True, inferSchema=True)
    drop_cols = ["application_id", "application_date"]
    df = raw_df.drop(*drop_cols)

    label_indexer = StringIndexer(inputCol="approval_status", outputCol="label")

    categorical_cols = [
        "employment_type",
        "marital_status",
        "education_level",
        "residence_type",
        "city_tier",
        "card_type_requested",
    ]
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in categorical_cols]
    encoder = OneHotEncoder(inputCols=[f"{col}_idx" for col in categorical_cols], outputCols=[f"{col}_vec" for col in categorical_cols], handleInvalid="keep")

    numeric_cols = ["age", "annual_income", "credit_score", "loan_amount", "years_employed", "debt_to_income_ratio", "existing_loans"]
    feature_cols = [f"{col}_vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

    classifier = SparkRF(labelCol="label", featuresCol="features", numTrees=50)
    pipeline = Pipeline(stages=[label_indexer] + indexers + [encoder, assembler, classifier])

    selected_cols = ["approval_status"] + categorical_cols + numeric_cols
    df = df.dropna(subset=selected_cols)

    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = binary_evaluator.evaluate(predictions)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = multi_evaluator.evaluate(predictions)

    rows = int(predictions.count())
    spark.stop()
    return {"rows": rows, "auc": float(auc), "accuracy": float(accuracy)}


def run_sklearn(path: str):
    df = pd.read_csv(path)
    # Drop ids and dates
    df = df.drop(columns=[col for col in ["application_id", "application_date"] if col in df.columns])

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=["approval_status"]) 

    categorical_cols = [
        "employment_type",
        "marital_status",
        "education_level",
        "residence_type",
        "city_tier",
        "card_type_requested",
    ]
    numeric_cols = ["age", "annual_income", "credit_score", "loan_amount", "years_employed", "debt_to_income_ratio", "existing_loans"]

    # Ensure numeric columns exist and fill na
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

    # One-hot encode categoricals (handle unseen by drop_first=False)
    df_cat = pd.get_dummies(df[categorical_cols].fillna("__MISSING__")) if any(col in df.columns for col in categorical_cols) else pd.DataFrame()

    X = pd.concat([df[numeric_cols].fillna(0), df_cat], axis=1)
    y = (df["approval_status"].astype(str).str.lower().isin(["approved", "accept", "yes"]) ).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    auc = roc_auc_score(y_test, probs) if probs is not None and len(set(y_test)) > 1 else float("nan")
    acc = accuracy_score(y_test, preds)

    return {"rows": len(X_test), "auc": float(auc), "accuracy": float(acc)}


def main():
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        print(f"Data file not found: {data_file.resolve()}")
        sys.exit(1)

    if USE_PYSPARK:
        try:
            metrics = run_pyspark(str(data_file))
            print(f"Test rows: {metrics['rows']}")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            print("PySpark failed to run; falling back to scikit-learn. Error:", e)
            metrics = run_sklearn(str(data_file))
            print(f"Test rows: {metrics['rows']}")
            print(f"AUC: {metrics['auc']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
    else:
        print("PySpark not available in this environment — using scikit-learn fallback.")
        metrics = run_sklearn(str(data_file))
        print(f"Test rows: {metrics['rows']}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
