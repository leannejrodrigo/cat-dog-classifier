import streamlit as st
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("transfer_model.keras")

# App title
st.title("Cat vs Dog Classifier")

# App description
st.write("Upload an image and the AI will predict whether it is a cat or a dog.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width=300)

    # Resize image to match training size
    resized_image = image.resize((128, 128))

    # Convert image to numbers
    img_array = tf.keras.utils.img_to_array(resized_image)

    # Add batch dimension
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)[0][0]

    # Convert to percentages
    dog_confidence = prediction * 100
    cat_confidence = (1 - prediction) * 100

    # Show results
    st.subheader("Result")

    if prediction > 0.5:
        st.success("Prediction: Dog")
    else:
        st.success("Prediction: Cat")

    st.write(f"Dog confidence: {dog_confidence:.2f}%")
    st.write(f"Cat confidence: {cat_confidence:.2f}%")
    