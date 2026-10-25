# Document Classification using NLP

Classifies news articles into 5 categories (business, entertainment, politics, sport, tech)
using an NLP pipeline and compares classical ML models (optionally DistilBERT).

## Project structure
```
data/bbc_news.csv          dataset (1490 articles)
src/preprocess.py          text cleaning (lowercase, stopwords, lemmatization)
src/train.py               trains + compares 4 models, saves plots/metrics/model
src/bert_finetune_colab.py optional DistilBERT fine-tuning (run on Colab)
app/app.py                 Streamlit demo
models/                    saved best model
results/                   metrics table, confusion matrix, error analysis
```

## How to run
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/app.py
```

## Pipeline
1. Clean text (lowercase, remove URLs/punctuation/stopwords, lemmatize)
2. TF-IDF features (unigrams + bigrams)
3. Models: Naive Bayes, Logistic Regression, Linear SVM, Random Forest
4. Evaluate: accuracy, precision, recall, F1, 5-fold CV, confusion matrix, error analysis

## Results
See `results/model_comparison.csv` (best: Linear SVM, ~97.6% accuracy / macro F1 on the 20% test split).
