import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("C:\\Users\\shree\\OneDrive\\Desktop\\Ticket Classification Project\\Ticket-Classification-Project\\data\\customer_support_tickets.csv")

# df.head() df.info() df.columns

df["ticket_text"] = (
    df["Ticket Subject"].astype(str) + " " +
    df["Ticket Description"].astype(str)
)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)   # it removes punctuation & numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["ticket_text"].apply(clean_text)

# Text Sample of cleaning the data/preprocessing
# print("\n Sample Ticket Text (Before & After Cleaning)\n")
# for i in range(3):  # print first 3 tickets
#     print(f"Ticket {i+1}")
#     print("ORIGINAL TEXT:")
#     print(df.loc[i, "ticket_text"])
#     print("\nCLEANED TEXT:")
#     print(df.loc[i, "clean_text"])
#     print("\n" + "-"*80 + "\n")

# Category Prediction Model
X = df["clean_text"]
y_category = df["Ticket Type"]
y_priority = df["Ticket Priority"]

X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
    X,
    y_category,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_category 
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Sanity Check
# print("TF-IDF Train Shape:", X_train_tfidf.shape)
# print("TF-IDF Test Shape:", X_test_tfidf.shape)

category_model = LogisticRegression(max_iter=1000, n_jobs=-1)
category_model.fit(X_train_tfidf, y_cat_train)
y_cat_pred = category_model.predict(X_test_tfidf)

print("\n--- Ticket Category Classification Results ---\n")

print("Accuracy:", accuracy_score(y_cat_test, y_cat_pred))
print("\nClassification Report:\n")
print(classification_report(y_cat_test, y_cat_pred))

cm = confusion_matrix(y_cat_test, y_cat_pred)
labels = category_model.classes_

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ticket Category")
plt.show()

"""
Business Insight:
-The model correctly classifies the majority of the ticket categories.
- Most of the misclassifications occur between similar type issues, for example, Account vs General Query.
It can go a long way in reducing manual triage effort.
"""

# Priority Classification Model
priority_model = LogisticRegression(max_iter = 1000, n_jobs = -1)
priority_model.fit(X_train_tfidf, y_pri_train)
y_pri_pred = priority_model.predict(X_test_tfidf)

print("\n--- Ticket Priority Classification Results ---\n")

print("Accuracy:", accuracy_score(y_pri_test, y_pri_pred))
print("\nClassification Report:\n")
print(classification_report(y_pri_test, y_pri_pred))

cm_pri = confusion_matrix(y_pri_test, y_pri_pred)
labels_pri = priority_model.classes_

plt.figure(figsize=(6, 5))
sns.heatmap(cm_pri, annot=True, fmt="d", xticklabels=labels_pri, yticklabels=labels_pri)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ticket Priority")
plt.show()

"""
Business Insight:
-The model correctly classifies the majority of the ticket categories.
- Most of the misclassifications occur between similar type issues, for example, Account vs General Query.
It can go a long way in reducing manual triage effort.
"""

# Class-wise performance summary
category_report = classification_report(
    y_cat_test, y_cat_pred, output_dict=True
)

print("\nCategory-wise F1 Scores:")
for label, metrics in category_report.items():
    if isinstance(metrics, dict):
        print(f"{label}: F1-score = {metrics['f1-score']:.2f}")

priority_report = classification_report(
    y_pri_test, y_pri_pred, output_dict=True
)

print("\nPriority-wise F1 Scores:")
for label, metrics in priority_report.items():
    if isinstance(metrics, dict):
        print(f"{label}: F1-score = {metrics['f1-score']:.2f}")

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(category_model, "models/category_model.pkl")
joblib.dump(priority_model, "models/priority_model.pkl")

print("\nModels and vectorizer saved successfully.")

# Demo Function
def predict_ticket(ticket_subject, ticket_description):
    text = ticket_subject + " " + ticket_description
    text_clean = clean_text(text)
    text_vec = tfidf.transform([text_clean])

    category = category_model.predict(text_vec)[0]
    priority = priority_model.predict(text_vec)[0]

    return category, priority

sample_category, sample_priority = predict_ticket(
    "Billing issue, lost payment",
    "I was charged twice and the payment did not go through"
)

print("\nSample Prediction:")
print("Predicted Category:", sample_category)
print("Predicted Priority:", sample_priority)