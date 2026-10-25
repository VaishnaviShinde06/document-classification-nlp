"""Streamlit demo: paste a document, get its predicted category."""
import sys, json
from pathlib import Path
import numpy as np
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from preprocess import clean_text  # noqa: E402

@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    meta = json.load(open(ROOT / "models" / "meta.json"))
    return model, meta

def predict(model, text):
    x = clean_text(text)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([x])[0]
    else:  # LinearSVC: softmax over decision scores
        s = model.decision_function([x])[0]
        e = np.exp(s - s.max()); probs = e / e.sum()
    return model.classes_, probs

st.set_page_config(page_title="Document Classifier", page_icon="📄")
st.title("📄 Document Classifier")
st.caption("NLP pipeline: text cleaning -> TF-IDF -> trained classifier")

model, meta = load()
st.sidebar.write(f"**Model:** {meta['best_model']}")
st.sidebar.write("**Categories:** " + ", ".join(meta["labels"]))

text = st.text_area("Paste an article or document:", height=220)
if st.button("Classify") and text.strip():
    classes, probs = predict(model, text)
    top = int(np.argmax(probs))
    st.success(f"Predicted category: **{classes[top]}** ({probs[top]*100:.1f}% confidence)")
    st.bar_chart({c: float(p) for c, p in zip(classes, probs)})
