"""Train and compare classical models; save metrics, plots and the best model."""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from preprocess import clean_text

df = pd.read_csv("data/bbc_news.csv").dropna()
df["clean"] = df["text"].apply(clean_text)

# EDA plots
plt.figure(figsize=(6, 4))
sns.countplot(y="label", data=df, order=df["label"].value_counts().index)
plt.title("Class distribution"); plt.tight_layout()
plt.savefig("results/class_distribution.png", dpi=150); plt.close()

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=5),  # calibrated => real probabilities
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
                     ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro")
    cv = cross_val_score(pipe, df["clean"], df["label"], cv=5, scoring="f1_macro").mean()
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": p, "Recall": r, "F1 (macro)": f1, "5-fold CV F1": cv})
    fitted[name] = (pipe, pred)
    print(f"\n=== {name} ===\n", classification_report(y_test, pred))

res = pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False).round(4)
res.to_csv("results/model_comparison.csv", index=False)
print("\n", res.to_string(index=False))

best = res.iloc[0]["Model"]
pipe, pred = fitted[best]
labels = sorted(df["label"].unique())
cm = confusion_matrix(y_test, pred, labels=labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion matrix - {best}"); plt.ylabel("True"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

# Error analysis: misclassified docs
test_idx = X_test.index
err = pd.DataFrame({"text": df.loc[test_idx, "text"].str[:300], "true": y_test.values, "pred": pred})
err = err[err["true"] != err["pred"]]
err.to_csv("results/misclassified_examples.csv", index=False)
print(f"\nMisclassified: {len(err)} of {len(y_test)}")

joblib.dump(pipe, "models/best_model.joblib")
json.dump({"best_model": best, "labels": labels}, open("models/meta.json", "w"))
print("Saved best model:", best)
