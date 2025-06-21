import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # ← Required to save CSV

# Load dataset
vocab_size = 10000


max_len = 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# Lets bulid the model
model = models.Sequential([
    layers.Embedding(vocab_size, 32, input_length=max_len),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training 
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# The following code is for the plotting
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#The following code predicts and saves for first 100 test samples
pred_probs = model.predict(x_test[:100])
pred_labels = (pred_probs > 0.5).astype(int)
true_labels = y_test[:100]

# Saveing the file
#results_df = pd.DataFrame({
    #'Predicted Probability': pred_probs.flatten(),
    #'Predicted Label': pred_labels.flatten(),
    #'True Label': true_labels
#})
#results_df.to_csv('imdb_predictions.csv', index=False)
#print("Saved results to imdb_predictions.csv")
