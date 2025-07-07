# NOMA AI
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('noma_model.keras')

# Define your classes
classes = [
    "Acne",
    "Actinic Keratosis",
    "Benign Tumors",
    "Bullous",
    "Candidiasis",
    "Drug Eruption",
    "Eczema",
    "Infestations/Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrheic Keratoses",
    "Skin Cancer",  # Malignant
    "Sun/Sunlight Damage",
    "Tinea",
    "Unknown/Normal",  # Normal
    "Vascular Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts"
]

# Define malignant and benign classes
malignant_classes = ["Skin Cancer"]
benign_classes = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis",
    "Drug Eruption", "Eczema", "Infestations/Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrheic Keratoses", 
    "Sun/Sunlight Damage", "Tinea", "Unknown/Normal", 
    "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
]

# Streamlit UI
st.title("NOMA AI🦠")
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6; /* Light blue color */
    }
    .section {
        background-image: url('https://images-provider.frontiersin.org/api/ipx/w=1200&f=png/https://www.frontiersin.org/files/Articles/965630/fphys-13-965630-HTML/image_m/fphys-13-965630-g001.jpg');
        background-size: cover; 
        background-repeat: no-repeat;
        background-position: center;
        padding: 60px; 
        border-radius: 10px;
        color: black; 
        margin: 20px 0;
        height: 400px; 
    }
    .sidebar .sidebar-content {
        background-color: #A020F0; 
        color: purple; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("Thank you for using this Skin Cancer AI Detection CNN")
st.write("CNNs are highly effective for detecting melanoma due to their capability to process image data. They excel in tasks like classification and object recognition, often surpassing human dermatologists in accuracy.")
st.markdown('</div>', unsafe_allow_html=True)

# Camera input
image_file = st.camera_input("Take a picture of the skin condition")

if image_file is not None:
    # Display the uploaded image
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = np.array(image.resize((224, 224))) / 255.0  # Adjust size and normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    if st.button("Classify"):
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_class = classes[class_index]

        # Display results
        if predicted_class in malignant_classes:
            st.subheader("Malignant")
        else:
            st.subheader("Benign")
        
        st.success(f"Predicted Class: {predicted_class}")
