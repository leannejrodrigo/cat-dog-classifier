import tensorflow as tf
import numpy as np

# Load saved model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Load image
img = tf.keras.utils.load_img(
    "test_images/test1.jpg",
    target_size=(128, 128)
)

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)

# Add extra dimension
img_array = tf.expand_dims(img_array, 0)

# Predict
prediction = model.predict(img_array)

print("Raw prediction:", prediction[0][0])

# Interpret prediction
if prediction[0][0] > 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")