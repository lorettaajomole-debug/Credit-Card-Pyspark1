import streamlit as st
from pathlib import Path

st.title("Credit Card Approval — Model Trainer")

DATA_PATH = "Credit_Card_Approval_10000_70_30.csv"

PRETRAINED_MODEL = Path("pretrained_model.joblib")
PRETRAINED_META = Path("pretrained_model_meta.json")

if PRETRAINED_MODEL.exists() and PRETRAINED_META.exists():
    import json
    import joblib
    try:
        meta = json.loads(PRETRAINED_META.read_text())
        st.success("Loaded pretrained model")
        st.metric("Rows (test)", meta.get("rows"))
        try:
            st.metric("AUC", f"{meta.get('auc', float('nan')):.4f}")
        except Exception:
            st.metric("AUC", meta.get('auc'))
        try:
            st.metric("Accuracy", f"{meta.get('accuracy', float('nan')):.4f}")
        except Exception:
            st.metric("Accuracy", meta.get('accuracy'))
        if st.button("Show raw metadata"):
            st.json(meta)
    except Exception as e:
        st.warning(f"Failed to load pretrained metadata: {e}")

try:
    from credit_card_classification_pyspark import run_pyspark, run_sklearn, USE_PYSPARK
except Exception as e:
    st.error(f"Failed importing training module: {e}")
    raise

st.markdown("Select backend and train the model.\n\nIf PySpark isn't available the app will automatically use scikit-learn.")

backend = st.selectbox("Backend", options=["auto", "pyspark", "sklearn"], index=0)

data_file = Path(DATA_PATH)

# Provide upload / URL fallback when dataset is not present in the deployed app
uploaded = None
download_url = None
if not data_file.exists():
    st.info("Data file not found in app folder. You can upload a CSV or provide a raw URL to the CSV.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    download_url = st.text_input("Or enter a direct URL to a CSV (optional)")

use_backend = backend
if backend == "auto":
    use_backend = "pyspark" if USE_PYSPARK else "sklearn"

def _prepare_data_path():
    # Returns a path-like object or string suitable for the training functions.
    if data_file.exists():
        return str(data_file)
    if uploaded is not None:
        # write uploaded file to temporary path (required by Spark)
        temp_path = Path("/tmp") / "uploaded_data.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())
        return str(temp_path)
    if download_url:
        import requests
        temp_path = Path("/tmp") / "downloaded_data.csv"
        try:
            resp = requests.get(download_url, timeout=30)
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(resp.content)
            return str(temp_path)
        except Exception as e:
            st.error(f"Failed to download CSV from URL: {e}")
            return None
    return None

if st.button("Train model"):
    data_path = _prepare_data_path()
    if not data_path:
        st.error("No dataset provided. Upload a CSV or provide a valid URL.")
    else:
        st.info(f"Using backend: {use_backend}")
        with st.spinner("Training — this can take a while on first run..."):
            try:
                if use_backend == "pyspark":
                    metrics = run_pyspark(data_path)
                else:
                    metrics = run_sklearn(data_path)
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)
            else:
                st.success("Training completed")
                st.metric("Rows (test)", metrics.get("rows"))
                try:
                    st.metric("AUC", f"{metrics.get('auc', float('nan')):.4f}")
                except Exception:
                    st.metric("AUC", metrics.get('auc'))
                try:
                    st.metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.4f}")
                except Exception:
                    st.metric("Accuracy", metrics.get('accuracy'))
                st.json(metrics)
