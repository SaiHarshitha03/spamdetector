📧 SMS Spam Detection using Naive Bayes

This is a simple machine learning project to classify SMS messages as spam or ham (not spam) using text preprocessing, TF-IDF vectorization, and a Naive Bayes classifier.


🚀 Features
Uses the SMS Spam Collection dataset
Text cleaning (lowercase, remove punctuation)
TF-IDF feature extraction
Multinomial Naive Bayes classification
Evaluation metrics: Accuracy, Precision, Recall, F1-score
Custom message prediction


📂 Dataset Used
Source: UCI SMS Spam Collection Dataset
Alternate GitHub URL:
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv


🛠️ Requirements
Install the required Python libraries:
pip install pandas scikit-learn

🧪 Example Output:
Accuracy: 0.98
Precision: 0.97
Recall: 0.94
F1 Score: 0.95

Spam
Not Spam

🧠 Model Used
Multinomial Naive Bayes: Best suited for word counts or TF-IDF scores in text classification.
🔍 Future Improvements:
Add more advanced preprocessing (stemming, stopword removal)
Try other models (SVM, Logistic Regression)
Deploy using Flask/Django for a web interface
Add input from WhatsApp, Telegram, or email
Use deep learning models like LSTM
Add GUI or web interface (e.g., Flask app)
Store prediction history

📜 License
This project is for educational purposes. Based on public datasets.


