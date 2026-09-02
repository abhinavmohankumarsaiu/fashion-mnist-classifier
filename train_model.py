import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)

# 1. Load Fashion-MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("Training images:", train_images.shape)
print("Test images:", test_images.shape)

# 2. Normalize to [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# 3. CNN requires an explicit channel dimension
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

print("CNN training shape:", train_images.shape)
print("CNN test shape:", test_images.shape)

# 4. Build CNN
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(10)  # logits
])

model.summary()

# 5. Compile
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# 6. Train
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=32,
    validation_split=0.10,
    verbose=2,
)

# 7. Evaluate
test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels,
    verbose=2,
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# 8. Save the trained model for deployment
MODEL_FILE = "fashion_cnn_model.keras"
model.save(MODEL_FILE)

print(f"Saved trained model as: {MODEL_FILE}")

