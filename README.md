# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and import required libraries.
2. Preprocess the data and convert labels into numerical values.
3. Convert text messages into feature vectors using TF-IDF Vectorization.
4. Train the Support Vector Machine model using training data.
5. Predict test data and evaluate the result using a confusion matrix.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Deepak K R
RegisterNumber:  212225040057 (25008695)
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# TF-IDF with slight tweak: including bigrams (ngram_range) to catch phrases
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM - switched to 'rbf' kernel for non-linear boundaries
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='mako',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("SVM Spam Detection Performance")
plt.show()

sample_msg = ["CONGRATULATIONS! You've won a $1000 Walmart gift card. Click here now!"]
sample_vec = vectorizer.transform(sample_msg)
prediction = "Spam" if model.predict(sample_vec)[0] == 1 else "Ham"
print(f"Sample Result: {prediction}")
```

## Output:
![alt text](<Screenshot 2026-03-10 111625.png>)
![alt text](<Screenshot 2026-03-10 111634.png>)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
