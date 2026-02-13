import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your saved model
model = tf.keras.models.load_model('my_cifar10_model.h5')
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("AI Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image (Step 2 from your PDF)
    img = image.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    
    st.write(f"## This looks like a: {result}")