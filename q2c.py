import numpy as np 

import pandas as pd 

import tensorflow as tf 

from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.model_selection import train_test_split, KFold 

from sklearn.preprocessing import LabelEncoder 

from sklearn.metrics import accuracy_score 

 

# Load the TREC dataset 

trec_data = pd.read_csv("/content/train.csv") 

 

# Inspect the data 

print("Columns in the dataset:", trec_data.columns) 

print("First few rows of the dataset:") 

print(trec_data.head()) 

 

# Assuming the correct column names are 'text' and 'label' 

# Update these names if they differ in your CSV file 

texts = trec_data['text'].values 

labels = trec_data["label-coarse"].values 

 

# Feature engineering: TF-IDF vectorization 

vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed 

X = vectorizer.fit_transform(texts).toarray() 

 

# Encode the labels 

label_encoder = LabelEncoder() 

y = label_encoder.fit_transform(labels) 

 

# Split data into training and test sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Define the neural network model 

model = tf.keras.Sequential([ 

    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]), 

    tf.keras.layers.Dropout(0.5), 

    tf.keras.layers.Dense(64, activation='relu'), 

    tf.keras.layers.Dropout(0.5), 

    tf.keras.layers.Dense(1, activation='sigmoid') 

]) 

 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

 

# Train the model 

model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1) 

 

# Evaluate the model 

loss, accuracy = model.evaluate(X_test, y_test) 

print(f"Test loss: {loss}, Test accuracy: {accuracy}") 

 

# Perform K-Fold Cross-Validation 

kf = KFold(n_splits=5, shuffle=True, random_state=42) 

fold_accuracies = [] 

 

for train_index, test_index in kf.split(X): 

    X_train_fold, X_val_fold = X[train_index], X[test_index] 

    y_train_fold, y_val_fold = y[train_index], y[test_index] 

     

    # Reinitialize the model for each fold to avoid weight leakage 

    fold_model = tf.keras.Sequential([ 

        tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]), 

        tf.keras.layers.Dropout(0.5), 

        tf.keras.layers.Dense(64, activation='relu'), 

        tf.keras.layers.Dropout(0.5), 

        tf.keras.layers.Dense(1, activation='sigmoid') 

    ]) 

    fold_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

     

    fold_model.fit(X_train_fold, y_train_fold, epochs=10, verbose=0) 

    y_pred = (fold_model.predict(X_val_fold) > 0.5).astype("int32") 

    accuracy = accuracy_score(y_val_fold, y_pred) 

    fold_accuracies.append(accuracy) 

 

print(f"K-Fold Cross-Validation accuracies: {fold_accuracies}") 

print(f"Average accuracy: {np.mean(fold_accuracies)}") 

 
