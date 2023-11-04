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