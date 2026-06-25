from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.master("local[*]").appName("CreditCardApprovalClassification").getOrCreate()

    data_path = "Credit_Card_Approval_10000_70_30.csv"
    raw_df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Drop columns that are not predictive or that should not be used directly
    drop_cols = ["application_id", "application_date"]
    df = raw_df.drop(*drop_cols)

    # Map approval status to a binary label
    label_indexer = StringIndexer(inputCol="approval_status", outputCol="label")

    categorical_cols = [
        "employment_type",
        "marital_status",
        "education_level",
        "residence_type",
        "city_tier",
        "card_type_requested",
    ]
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{col}_idx" for col in categorical_cols],
        outputCols=[f"{col}_vec" for col in categorical_cols],
        handleInvalid="keep",
    )

    numeric_cols = [
        "age",
        "annual_income",
        "credit_score",
        "loan_amount",
        "years_employed",
        "debt_to_income_ratio",
        "existing_loans",
    ]

    feature_cols = [f"{col}_vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

    classifier = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=50)

    pipeline = Pipeline(stages=[label_indexer] + indexers + [encoder, assembler, classifier])

    # Drop rows with nulls in any of the selected feature columns
    selected_cols = ["approval_status"] + categorical_cols + numeric_cols
    df = df.dropna(subset=selected_cols)

    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = binary_evaluator.evaluate(predictions)

    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = multi_evaluator.evaluate(predictions)

    print(f"Test rows: {predictions.count()}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()
