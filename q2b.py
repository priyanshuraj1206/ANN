import tensorflow as tf 

from sklearn.datasets import load_iris 

from sklearn.model_selection import train_test_split, KFold 

from sklearn.preprocessing import OneHotEncoder 

from sklearn.metrics import accuracy_score 

 

# Load IRIS dataset 

iris = load_iris() 

X, y = iris.data, iris.target 

 

# One-hot encode the target labels 

encoder = OneHotEncoder(sparse=False) 

y = encoder.fit_transform(y.reshape(-1, 1)) 

 

# Split data into training and test sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Define the neural network model 

model = tf.keras.Sequential([ 

    tf.keras.layers.Dense(10, activation='relu', input_dim=X_train.shape[1]), 

    tf.keras.layers.Dense(10, activation='relu'), 

    tf.keras.layers.Dense(y_train.shape[1], activation='softmax') 

]) 

 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

 

# Train the model 

model.fit(X_train, y_train, epochs=50, verbose=1) 

 

# Evaluate the model 

loss, accuracy = model.evaluate(X_test, y_test) 

print(f"Test loss: {loss}, Test accuracy: {accuracy}") 

 

# Perform K-Fold Cross-Validation 

kf = KFold(n_splits=5, shuffle=True, random_state=42) 

fold_accuracies = [] 

 

for train_index, test_index in kf.split(X): 

    X_train, X_val = X[train_index], X[test_index] 

    y_train, y_val = y[train_index], y[test_index] 

     

    model.fit(X_train, y_train, epochs=50, verbose=0) 

    y_pred = np.argmax(model.predict(X_val), axis=1) 

    y_true = np.argmax(y_val, axis=1) 

    accuracy = accuracy_score(y_true, y_pred) 

    fold_accuracies.append(accuracy) 

 

print(f"K-Fold Cross-Validation accuracies: {fold_accuracies}") 

print(f"Average accuracy: {np.mean(fold_accuracies)}") 

 
