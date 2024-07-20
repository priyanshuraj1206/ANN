import tensorflow as tf 

from sklearn.datasets import load_iris 

from sklearn.preprocessing import StandardScaler, OneHotEncoder 

from sklearn.model_selection import train_test_split 

 

# Load and preprocess the IRIS dataset 

iris = load_iris() 

X = iris.data 

y = iris.target 

 

# Standardize features 

scaler = StandardScaler() 

X = scaler.fit_transform(X) 

 

# One-hot encode the labels 

encoder = OneHotEncoder(sparse=False) 

y = encoder.fit_transform(y.reshape(-1, 1)) 

 

# Split the dataset 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Define the FFNN model 

model = tf.keras.Sequential([ 

    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 

    tf.keras.layers.Dropout(0.5), 

    tf.keras.layers.Dense(32, activation='relu'), 

    tf.keras.layers.Dropout(0.5), 

    tf.keras.layers.Dense(3, activation='softmax') 

]) 

 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

 

# Train the model 

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1) 

 

# Evaluate the model 

loss, accuracy = model.evaluate(X_test, y_test) 

print(f"Test loss: {loss}, Test accuracy: {accuracy}") 

 

# Plotting learning curves 

import matplotlib.pyplot as plt 

 

plt.plot(history.history['accuracy'], label='accuracy') 

plt.plot(history.history['val_accuracy'], label = 'val_accuracy') 

plt.xlabel('Epoch') 

plt.ylabel('Accuracy') 

plt.ylim([0, 1]) 

plt.legend(loc='lower right') 

plt.show() 

 
