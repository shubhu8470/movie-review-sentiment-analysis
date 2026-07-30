import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ===============================
# Dataset
# ===============================

positive_reviews = [
    "Everyone agreed the ending was excellent",
    "I found the script to be wonderful",
    "The show turned out to be surprisingly brilliant",
    "I really loved how wonderful the performance was",
    "Everyone agreed the movie was fantastic",
    "I really loved how fantastic the direction was",
    "Everyone agreed this film was lovely",
    "I found the movie to be beautiful",
    "The acting was fun",
    "Everyone agreed the performance was beautiful",
    "The movie was impressive",
    "What a excellent show",
    "I found the direction to be excellent",
    "The performance was truly superb from start to finish",
    "I really loved how beautiful this film was",
    "I found the ending to be brilliant",
    "Everyone agreed the screenplay was refreshing",
    "The ending was engaging and kept me hooked",
    "I really loved how superb the ending was",
    "I really loved how stunning the story was",
    "What a beautiful film",
    "The show turned out to be surprisingly great",
    "What a engaging series",
    "I found the performance to be fantastic",
    "The show was truly heartwarming from start to finish",
    "The ending was excellent and kept me hooked",
    "I found the direction to be wonderful",
    "The show turned out to be surprisingly beautiful",
    "The ending turned out to be surprisingly stunning",
    "The performance was great and kept me hooked",
    "What a fantastic film",
    "I found the screenplay to be stunning",
    "What a solid movie",
    "The script was beautiful and kept me hooked",
    "Everyone agreed the plot was stunning",
    "The script was captivating",
    "The screenplay was truly captivating from start to finish",
    "The performance was brilliant and kept me hooked",
    "What a delightful movie",
    "I really loved how solid the story was",
    "The direction was remarkable and kept me hooked",
    "This film was outstanding and kept me hooked",
    "What a lovely direction",
    "What a heartwarming story",
    "This series turned out to be surprisingly heartwarming",
    "I really loved how remarkable the script was",
    "The story was truly fantastic from start to finish",
    "I really loved how impressive the story was",
    "The movie was truly great from start to finish",
    "The plot was superb",
    "I absolutely loved this movie",
    "The acting was excellent",
    "This was a wonderful movie"
]

negative_reviews = [
    "The story was dull and I lost interest quickly",
    "I found the performance to be tiresome",
    "The show turned out to be surprisingly terrible",
    "The script was poor and I lost interest quickly",
    "This film was unpleasant",
    "What a tedious acting",
    "I found the movie to be disgusting",
    "The show was frustrating",
    "This film turned out to be surprisingly frustrating",
    "I really disliked how poor the performance was",
    "This series was forgettable",
    "The show was unpleasant",
    "What a unconvincing show",
    "The show was tedious",
    "I found the story to be underwhelming",
    "I really disliked how sloppy this film was",
    "I really disliked how boring the direction was",
    "I found the script to be mediocre",
    "The story was truly forgettable",
    "The direction was truly bad",
    "I found the screenplay to be boring",
    "What a horrible film",
    "The plot was terrible",
    "The acting was poor",
    "The story was lifeless",
    "The direction was horrible",
    "The movie was bad",
    "I hated this movie",
    "This film was terrible and boring",
    "This was a complete waste of time"
]

reviews = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

dataset = pd.DataFrame({
    "review": reviews,
    "sentiment": labels
})

print(dataset.head())
print("\nDataset Shape:", dataset.shape)

# ===============================
# Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    dataset["review"],
    dataset["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=dataset["sentiment"]
)

# ===============================
# TF-IDF
# ===============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ===============================
# Model
# ===============================

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

# ===============================
# Confusion Matrix
# ===============================

cm = confusion_matrix(y_test, predictions)

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()

# ===============================
# Save Model
# ===============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel saved successfully!")
print("Location: models/sentiment_model.pkl")
print("Location: models/vectorizer.pkl")

# ===============================
# Test Prediction
# ===============================

sample = "The movie was absolutely fantastic."

sample_vector = vectorizer.transform([sample])

prediction = model.predict(sample_vector)[0]
probability = model.predict_proba(sample_vector).max()

if prediction == 1:
    sentiment = "Positive 😊"
else:
    sentiment = "Negative 😞"

print("\nSample Review:", sample)
print("Prediction:", sentiment)
print("Confidence:", round(probability * 100, 2), "%")