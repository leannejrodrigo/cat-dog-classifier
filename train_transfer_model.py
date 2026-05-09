import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

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

# Load pretrained MobileNetV2
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze pretrained layers
base_model.trainable = False

# Build transfer learning model
model = models.Sequential([

    layers.Rescaling(1./255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation='relu'),

    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=3
)

# Evaluate
loss, accuracy = model.evaluate(test_dataset)

print("Accuracy:", accuracy)

# Save model
model.save("transfer_model.keras")

print("Transfer learning model saved!")