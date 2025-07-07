# NOMA AI
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model(r'C:\Users\havil\Downloads\NOMA\NOMA-AI\noma_model.keras')

# Define your classes
classes = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
    "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
    "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea",
    "Seborrheic Keratoses", "Skin Cancer",  # Malignant
    "Sun/Sunlight Damage", "Tinea", "Unknown/Normal",  # Normal
    "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
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

# Patient Information Section
st.header("Patient Information")

# Tabular Inputs
age = st.slider("Age", 1, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
skin_tone = st.selectbox("Skin Tone (Fitzpatrick Scale)", ["I", "II", "III", "IV", "V", "VI"])
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

        # Combine tabular input for prediction
        tabular_input = preprocess_tabular(age, gender, skin_tone, location, itching, bleeding, duration)

        # Display results
        if predicted_class in malignant_classes:
            st.subheader("Malignant")
        else:
            st.subheader("Benign")
        
        st.success(f"Predicted Class: {predicted_class}")


# Function to collect feedback
def collect_feedback():
    st.title(":rainbow[Feedback] Form📖")
    
    # Add a text area for the feedback
    feedback = st.text_area("Please share your feedback to improve this app💕", "", height=150)
    
    # Add a submit button
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback🫶!")
            # Save feedback to a file, database, or send it via email, etc.
            save_feedback(feedback)
        else:
            st.error("Please enter some feedback before submitting😡.")

# Function to save feedback (can be customized to store feedback anywhere)
def save_feedback(feedback):
    # Example: Save to a text file (or database)
    with open("user_feedback.txt", "a") as f:
        f.write(f"Feedback: {feedback}\n{'-'*50}\n")
    st.info("Your feedback has been recorded.")

# Show the feedback form
collect_feedback()
