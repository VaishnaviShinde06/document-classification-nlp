"""Optional advanced step. Run on Google Colab (free GPU):
    !pip install transformers datasets torch scikit-learn pandas accelerate
Upload data/bbc_news.csv to Colab first.
"""
import pandas as pd, numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

df = pd.read_csv("bbc_news.csv").dropna()
labels = sorted(df["label"].unique())
l2i = {l: i for i, l in enumerate(labels)}
df["y"] = df["label"].map(l2i)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def enc(b): return tok(b["text"], truncation=True, padding="max_length", max_length=256)
ds_train = Dataset.from_pandas(train_df[["text", "y"]].rename(columns={"y": "labels"})).map(enc, batched=True)
ds_test = Dataset.from_pandas(test_df[["text", "y"]].rename(columns={"y": "labels"})).map(enc, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(labels),
    id2label=dict(enumerate(labels)), label2id=l2i)

def metrics(p):
    pred = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "f1_macro": f1_score(p.label_ids, pred, average="macro")}

args = TrainingArguments("bert_out", num_train_epochs=3, per_device_train_batch_size=16,
                         per_device_eval_batch_size=32, eval_strategy="epoch",
                         learning_rate=2e-5, weight_decay=0.01, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=ds_train,
                  eval_dataset=ds_test, compute_metrics=metrics)
trainer.train()
out = trainer.predict(ds_test)
pred = np.argmax(out.predictions, axis=1)
print(classification_report(out.label_ids, pred, target_names=labels))
