# Fake News Detection using SVM (TF-IDF + LinearSVC)

A binary text classification project that detects fake vs. real news articles using TF-IDF vectorization and a Linear Support Vector Classifier (SVM). Beyond just training a model, this project includes a full investigation into data leakage and model generalization — documented honestly, including where the model struggles.

🔗 **Live demo:** [Streamlit App Link] *(add after deployment)*

---

## Dataset

- **Source:** [`mucahiddemircan/real-and-fake-news-dataset`](https://www.kaggle.com/) on Kaggle
- **Size:** 45,757 articles, balanced classes (~50/50 real vs. fake)
- **Note on recency:** Despite the Kaggle listing being "recently updated," the article *content* dates back to 2015–2017 (verified by checking year-mention frequency in the text). This has real implications — see Limitations below.

---

## Approach

1. **EDA** — inspected nulls/duplicates (none found), generated word clouds for real vs. fake news to visually compare vocabulary patterns.
2. **First-pass model** — TF-IDF (max_features=1000, English stop words removed) + LinearSVC pipeline. Achieved **97.69% accuracy**.
3. **Data leakage investigation** — suspicious of the unusually high accuracy, checked for shortcut signals.
4. **Cleaning & retraining** — removed the identified leakage pattern, retrained.
5. **Final evaluation** — confusion matrix + classification report.
6. **Real-world testing** — validated on genuinely external articles (both same-era and current) to test actual generalization, not just test-set accuracy.

---

## Data Leakage Investigation

Before trusting the 97.69% accuracy, I checked whether the model was learning genuine content patterns or exploiting a shortcut.

**Finding:** The word "Reuters" appeared in **94.9%** of real news articles vs. only **1.35%** of fake ones — a strong artifact of wire-service datelines (e.g. *"WASHINGTON (Reuters) -"*), not genuine content signal. A single-feature model using *only* "does the text contain 'Reuters'?" scored **96.84% accuracy** on its own — nearly matching the full model.

**Action taken:** Wrote a regex to strip dateline/byline patterns (e.g. `"WASHINGTON (Reuters) -"`, photo credits like `"REUTERS/Jason Lee"`) while preserving legitimate in-text mentions (e.g. *"...told Reuters..."*, which is genuine journalistic content, not a leakage artifact).

**Result after cleaning:** Accuracy remained stable at **97.66%** (97.69% → 97.66%), confirming the model relies on genuine content patterns across many features, not primarily this one shortcut.

---

## Final Results

| Metric | Score |
|---|---|
| Accuracy | 97.66% |
| Precision (avg) | 0.98 |
| Recall (avg) | 0.98 |
| F1-score (avg) | 0.98 |

Errors are balanced across both classes — no systematic bias toward either label.

---

## Real-World Testing & Limitations

Model was tested on genuinely external articles beyond the original dataset:

- **Same-era validation (ISOT Fake News Dataset, 2015–2017):** tested against real/fake articles from the same time period as training data — predictions were largely accurate, confirming the model performs reliably *within its training domain*.
- **Current-events testing (2026 articles from Al Jazeera & PolitiFact):** performance degraded — a genuine current-events real article was misclassified as fake, with low-confidence prediction scores (near 0), while a PolitiFact-confirmed fake article was correctly caught.

**Key limitation — Concept Drift:** The model was trained on 2015–2017 vocabulary and writing patterns. On unfamiliar recent topics/vocabulary, TF-IDF has no learned signal to draw on, leading to unstable, low-confidence predictions. This is a known real-world ML challenge, not a flaw specific to this implementation — it reflects why production models require periodic retraining on fresh data.

**Secondary limitation — Style vs. Substance:** The model appears to rely on writing style/tone patterns (formal wire-service tone vs. sensational tabloid tone) rather than verifying factual accuracy. A well-written, formally-toned fabricated article can be misclassified as real.

---

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (TfidfVectorizer, LinearSVC, Pipeline)
- WordCloud, Matplotlib (EDA visualization)
- Streamlit (deployment)
- joblib (model persistence)

---

## How to Run Locally

```bash
git clone [repo-link]
cd [repo-name]
pip install -r requirements.txt
streamlit run app.py
```

---

## Future Work

- Hyperparameter tuning via GridSearchCV (C, max_features) — not performed due to time constraints; current parameters are reasonable defaults, not optimized
- Periodic retraining with recent news sources to address concept drift
- Testing transformer-based models (e.g. DistilBERT) for better generalization to unfamiliar vocabulary/topics
- Incorporating fact-verification signals rather than relying solely on stylistic patterns

---

## Author

Maria Anwar