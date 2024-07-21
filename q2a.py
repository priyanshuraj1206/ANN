import numpy as np 

import pandas as pd 

import tensorflow as tf 

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.model_selection import train_test_split, KFold 

from sklearn.preprocessing import LabelEncoder 

from sklearn.metrics import accuracy_score 

 

# Example tweets and their sentiments (1 = positive, 0 = negative) 

tweets = [ 

    "I love this product! It is amazing.", 

    "I hate this product. It is terrible.", 

    "This is the best day ever!", 

    "This is the worst experience I've had." 

] 

sentiments = [1, 0, 1, 0]  # Example sentiment labels 

 

# Feature engineering 

vectorizer = CountVectorizer(binary=True) 

X = vectorizer.fit_transform(tweets).toarray() 

y = np.array(sentiments) 

 

# Split data into training and test sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Define the neural network model 

model = tf.keras.Sequential([ 

    tf.keras.layers.Dense(10, activation='relu', input_dim=X_train.shape[1]), 

    tf.keras.layers.Dense(1, activation='sigmoid') 

]) 

 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

 

# Train the model 

model.fit(X_train, y_train, epochs=10, verbose=1) 

 

# Evaluate the model 

loss, accuracy = model.evaluate(X_test, y_test) 

print(f"Test loss: {loss}, Test accuracy: {accuracy}") 

 

# Perform K-Fold Cross-Validation 

kf = KFold(n_splits=4, shuffle=True, random_state=42) 

fold_accuracies = [] 

 

for train_index, test_index in kf.split(X): 

    X_train, X_val = X[train_index], X[test_index] 

    y_train, y_val = y[train_index], y[test_index] 

     

    model.fit(X_train, y_train, epochs=10, verbose=0) 

    y_pred = (model.predict(X_val) > 0.5).astype("int32") 

    accuracy = accuracy_score(y_val, y_pred) 

    fold_accuracies.append(accuracy) 

 

print(f"K-Fold Cross-Validation accuracies: {fold_accuracies}") 

print(f"Average accuracy: {np.mean(fold_accuracies)}") 

 

 
