import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

X_clean = data['message'].apply(clean_text)
y = data['label'].map({'ham': 0, 'spam': 1})


# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_clean)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict function
def predict_spam(text):
    cleaned_text = clean_text(text)
    vec = vectorizer.transform([cleaned_text])
    pred = model.predict(vec)
    print("Spam" if pred[0] == 1 else "Not Spam")

predict_spam("Congratulations! You've won a free ticket!")
predict_spam("Hi, are we meeting tomorrow?")
