import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build the Neural Network
model = models.Sequential([
  layers.Flatten(input_shape=(28, 28)),     # Flatten 28x28 into 784
  layers.Dense(128, activation='relu'),     # Hidden layer with 128 neurons
  layers.Dropout(0.2),                      # Dropout to prevent overfitting
  layers.Dense(10, activation='softmax')    # Output layer (10 digits)
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
print("Training model...")
model.fit(x_train, y_train, epochs=5)

# 5. Evaluate accuracy on the test set
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")
