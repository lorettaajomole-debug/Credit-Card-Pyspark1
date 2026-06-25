Credit Card PySpark Classification
=================================

This repository contains a small script that trains a credit-approval classifier.

Key points:
- The script prefers PySpark when available, but falls back to pandas + scikit-learn when PySpark is not installed (useful for Streamlit cloud deployments that cannot install Java/PySpark).
- To run locally with PySpark, install PySpark into your virtual environment and ensure Java (JDK 17) is available.

Quick start (local, PySpark):

```bash
# activate your venv
source .venv/bin/activate
pip install pyspark numpy pandas
export JAVA_HOME=/path/to/jdk17
python credit_card_classification_pyspark.py
```

Quick start (Streamlit / lightweight):

```bash
pip install -r requirements.txt
python credit_card_classification_pyspark.py
```

If you want to run a Streamlit app that imports this module, keep in mind that PySpark may not be available on the deployment platform; the script will automatically use scikit-learn instead.
