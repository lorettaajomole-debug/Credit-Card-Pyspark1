import streamlit as st
from pathlib import Path

st.title("Credit Card Approval — Model Trainer")

DATA_PATH = "Credit_Card_Approval_10000_70_30.csv"

try:
    from credit_card_classification_pyspark import run_pyspark, run_sklearn, USE_PYSPARK
except Exception as e:
    st.error(f"Failed importing training module: {e}")
    raise

st.markdown("Select backend and train the model.\n\nIf PySpark isn't available the app will automatically use scikit-learn.")

backend = st.selectbox("Backend", options=["auto", "pyspark", "sklearn"], index=0)

data_file = Path(DATA_PATH)
if not data_file.exists():
    st.error(f"Data file not found in app folder: {data_file.resolve()}")

use_backend = backend
if backend == "auto":
    use_backend = "pyspark" if USE_PYSPARK else "sklearn"

if st.button("Train model"):
    st.info(f"Using backend: {use_backend}")
    with st.spinner("Training — this can take a while on first run..."):
        try:
            if use_backend == "pyspark":
                metrics = run_pyspark(str(data_file))
            else:
                metrics = run_sklearn(str(data_file))
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)
        else:
            st.success("Training completed")
            st.metric("Rows (test)", metrics.get("rows"))
            st.metric("AUC", f"{metrics.get('auc', float('nan')):.4f}")
            st.metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.4f}")
            st.json(metrics)
