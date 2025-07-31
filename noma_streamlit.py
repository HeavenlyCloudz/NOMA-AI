# NOMA AI
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('new_noma_model.keras')

# Define your classes
classes = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
    "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
    "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea",
    "Seborrheic Keratoses", "Melanoma",  # Malignant
    "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Sun/Sunlight Damage",
    "Tinea", "Normal", "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
]

# Define malignant, benign, and normal classes
malignant_classes = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]
benign_classes = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis",
    "Drug Eruption", "Eczema", "Infestations/Bites", "Lichen", "Lupus",
    "Moles", "Psoriasis", "Rosacea", "Seborrheic Keratoses",
    "Sun/Sunlight Damage", "Tinea", "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
]
normal_classes = ["Normal"]

# Streamlit UI
st.title("NOMA AI🦠")
st.markdown("""
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
   """, unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("Thank you for using NOMA AI🩹")
st.write("CNNs are highly effective for detecting melanoma due to their capability to process image data. They excel in tasks like classification and object recognition, often surpassing human dermatologists in accuracy.")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("Visit my [GitHub](https://github.com/HeavenlyCloudz/NOMA-AI.git) repository for insight on my code.")

# Patient Information Section
st.header("Patient Information📋")

# Tabular Inputs
age = st.slider("Age", 1, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
skin_tone = st.selectbox("Skin Tone (Fitzpatrick Scale)", ["I", "II", "III", "IV", "V", "VI"])
st.write("Type I: Very light skin, Type II: Light skin, Type III: Medium skin, Type IV: Olive skin, Type V: Brown skin, Type VI: Very dark skin.")
location = st.selectbox("Lesion Location", ["Face", "Arm", "Leg", "Back", "Chest", "Other"])
itching = st.checkbox("Itching?")
bleeding = st.checkbox("Bleeding?")
duration = st.slider("Duration of Lesion (days)", 0, 365, 30)

# Convert categorical features to numerical or one-hot
def preprocess_tabular(age, gender, skin_tone, location, itching, bleeding, duration):
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    skin_map = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5}
    location_map = {"Face": 0, "Arm": 1, "Leg": 2, "Back": 3, "Chest": 4, "Other": 5}

    return np.array([
        age,
        gender_map[gender],
        skin_map[skin_tone],
        location_map[location],
        int(itching),
        int(bleeding),
        duration
    ])

# Function to assess likelihood based on tabular data
def assess_likelihood(tabular_input, predictions):
    likelihoods = {}
    for cls, prediction in zip(classes, predictions[0]):
        base_likelihood = prediction
        if tabular_input[0] > 50:  # Older age increases risk for some conditions
            base_likelihood *= 1.1
        if tabular_input[4]:  # Itching increases risk for certain conditions
            if cls in ["Eczema", "Psoriasis"]:
                base_likelihood *= 1.2
        likelihoods[cls] = min(base_likelihood, 1.0)  # Cap at 1.0
    return likelihoods

# Function to preprocess the image
def preprocess_image(image):
    img_array = np.array(image.resize((224, 224))) / 255.0  # Adjust size and normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function for Grad-CAM
def generate_gradcam(model, img_array, class_index):
    # Ensure that the image shape is correct
    if img_array.shape != (1, 224, 224, 3):
        raise ValueError("Image array must have shape (1, 224, 224, 3)")

    # Use the DenseNet output directly
    last_conv_layer = model.get_layer('densenet121')  # Adjust if needed

    # Create a model that maps the input image to the activations of the DenseNet output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Compute the gradient of the class output with respect to the feature map
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]

    # Pool the gradients across the channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Weight the output feature map by the pooled gradients
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap

# Camera input
image_file = st.camera_input("Take a picture of the skin condition")

if image_file is not None:
    # Display the uploaded image
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    # Make predictions
    if st.button("Classify"):
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_class = classes[class_index]

        # Generate Grad-CAM
        heatmap = generate_gradcam(model, img_array, class_index)

        # Create the Grad-CAM image
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))  # Corrected
        heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap
        heatmap = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)  # Overlay

        # Display Grad-CAM result
        st.image(heatmap, caption='Grad-CAM', use_container_width=True)

        # Combine tabular input for prediction
        tabular_input = preprocess_tabular(age, gender, skin_tone, location, itching, bleeding, duration)

        # Assess likelihoods
        likelihoods = assess_likelihood(tabular_input, predictions)

        # Display results
        if predicted_class in malignant_classes:
            st.subheader("Malignant")
        elif predicted_class in benign_classes:
            st.subheader("Benign")
        elif predicted_class in normal_classes:
            st.subheader("Normal")

        st.success(f"Predicted Class: {predicted_class}")

        # Display likelihood results
        st.subheader("Likelihood of Conditions:")
        for cls, likelihood in likelihoods.items():
            st.write(f"{cls}: {likelihood * 100:.2f}%")

# Function to collect feedback
def collect_feedback():
    st.title(":rainbow[Feedback] Form📖")

    # Add a text area for the feedback
    feedback = st.text_area("Please share your feedback to improve this app💕", "", height=150)

    # Add a submit button
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback🫶!")
            save_feedback(feedback)
        else:
            st.error("Please enter some feedback before submitting😡.")

# Function to save feedback (can be customized to store feedback anywhere)
def save_feedback(feedback):
    with open("user_feedback.txt", "a") as f:
        f.write(f"Feedback: {feedback}\n{'-'*50}\n")
    st.info("Your feedback has been recorded.")

# Show the feedback form
collect_feedback()
