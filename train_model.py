import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128, 128),
    batch_size=32
)

# Split dataset
train_size = int(len(dataset) * 0.8)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Build model
model = models.Sequential([

    layers.Rescaling(1./255),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

# Configure learning
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5
)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)

print("Accuracy:", accuracy)

# Save model
model.save("cat_dog_model.keras")

print("Model saved successfully!")