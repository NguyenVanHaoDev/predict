# train_product_classifier.py

import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

print("✅ Importing libraries...")

# === 1. Load & Preprocess Dataset ===
df = pd.read_csv('../data/product.csv', header=None, skiprows=1)
df.columns = ['text', 'product_code']
print("✅ Loaded dataset:")
print(df.head())

# === 2. Feature Extraction: TF-IDF ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df["text"])
y = df["product_code"]

# === 3. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# === 4. Grid Search to Tune Model ===
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'criterion': ['gini', 'entropy']
}
print("🔍 Running grid search for best parameters...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("✅ Best parameters:", grid_search.best_params_)

# === 5. Evaluation ===
y_pred = best_model.predict(X_test)
print("🎯 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# === 6. Save Model & Vectorizer ===
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/product_classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Model and vectorizer saved to /model")

# === 7. Sample Prediction (Optional) ===
sample_text = " chăn"
X_sample = vectorizer.transform([sample_text])
predicted = best_model.predict(X_sample)[0]
print(f"🧠 Dự đoán cho \"{sample_text}\": {predicted}")
