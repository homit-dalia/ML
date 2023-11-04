import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
import numpy as np
import pandas as pd

learning_rate = 0.001

custom_optimizer = Adam(learning_rate=learning_rate)

np.random.seed(0)
tf.random.set_seed(0)

from tensorflow import keras

model = keras.Sequential([
    
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.01),

    keras.layers.Dense(5, activation='softmax')
])

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Compile model
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit model on training data
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Model evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Calculate the classification report
y_pred = model.predict(X_test)
y_pred_classes = [int(i.argmax()) for i in y_pred]
class_report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)

print("Classification Report:")
print(class_report)



####################################################

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the CSV dataset
data = pd.read_csv('iris.csv')

# Split the dataset into features (X) and target labels (y)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Encode the target labels into numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a deep learning model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test data
y_pred = model.predict(X_test)
y_pred_classes = [np.argmax(pred) for pred in y_pred]

# Decode the numerical predictions back to labels
y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

# Generate a classification report
class_report = classification_report(y_test, y_pred_labels)
print("Classification Report:")
print(class_report)
