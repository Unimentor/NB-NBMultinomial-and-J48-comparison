import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nltk
from nltk.corpus import stopwords

path = "C:\\Users\\mahmo\\Desktop\\Abdullah Zafar\\eziline workspace\\"
data = pd.read_csv(path + "emails.csv")
X = data['text']
y = data['label']

# Preprocessing data
stop_words=stopwords.words('english')
def preprocess_text(text):
    text=text.lower()
    words=text.split()
    filtered_words=[i for i in words if i not in stop_words]
    return " ".join(filtered_words)
data['cleaned_text']=data['text'].apply(preprocess_text)

# Spliting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorizing text 
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Feature selection algorithm
def best_first_feature_selection(X_train, X_test, Y_train, Y_test, model, max_features=15):
    num_features = X_train.shape[1]
    selected_features = []
    remaining_features = list(range(num_features))
    best_accuracy = 0
    while remaining_features:
        best_feature = None
        best_accuracy_this_round = 0
        for feature in remaining_features:
            selected_features.append(feature)
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            model.fit(X_train_selected, Y_train)
            Y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(Y_test, Y_pred)
            
            if accuracy > best_accuracy_this_round:
                best_accuracy_this_round = accuracy
                best_feature = feature
            selected_features.remove(feature)
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"Feature {best_feature} selected, accuracy: {best_accuracy_this_round}")
        
        if max_features and len(selected_features) >= max_features:
            break
    
    return selected_features

# Training data
nb_model = MultinomialNB()
bnb_model = BernoulliNB()
j48_model = DecisionTreeClassifier(criterion='entropy')
models = [("Multinomial Naive Bayes", nb_model), ("Bernoulli Naive Bayes", bnb_model), ("J48 (Decision Tree)", j48_model)]
results = {}

for model_name, model in models:
    
    selected_features = best_first_feature_selection(X_train_vec, X_test_vec, y_train, y_test, model)
    X_train_selected = X_train_vec[:, selected_features]
    X_test_selected = X_test_vec[:, selected_features]
    
    start_time = time.time()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy
    end_time = time.time()
    
    results[model_name] = {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'time': end_time - start_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

# Print results and display confusion matrix
for model_name, result in results.items():
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {result['accuracy'] * 100:.2f}%")
    print(f"Error Rate: {result['error_rate'] * 100:.2f}%")
    print(f"Time Taken: {result['time']:.4f} seconds")
    print("Classification Report:\n", result['classification_report'])
    
    cm = result['confusion_matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Comparing accuracy and error rate
accuracy_comparison = {model_name: result['accuracy'] for model_name, result in results.items()}
error_rate_comparison = {model_name: result['error_rate'] for model_name, result in results.items()}

print("\nAlgorithm Accuracy Comparison:")
for model_name, accuracy in accuracy_comparison.items():
    print(f"{model_name}: {accuracy * 100:.2f}%")

print("\nAlgorithm Error Rate Comparison:")
for model_name, error_rate in error_rate_comparison.items():
    print(f"{model_name}: {error_rate * 100:.2f}%")
