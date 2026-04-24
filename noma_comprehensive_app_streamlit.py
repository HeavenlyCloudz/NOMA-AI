import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import hashlib
import io

# Page config
st.set_page_config(
    page_title="NOMA AI - Clinical Tracker",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .risk-high {
        background-color: #ff4757;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #ffa502;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #26de81;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
    }
    .info-box {
        background-color: #f1f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .timeline-item {
        border-left: 3px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        background: #f8f9fa;
        border-radius: 0 10px 10px 0;
    }
    .heatmap-container {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for tracking
if 'patients' not in st.session_state:
    st.session_state.patients = {}
if 'current_patient_id' not in st.session_state:
    st.session_state.current_patient_id = None
if 'assessments' not in st.session_state:
    st.session_state.assessments = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

# ==================== USER AUTHENTICATION SYSTEM ====================

# File to store user credentials
USER_DATA_FILE = "users.json"

def hash_password(password):
    """Hash a password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_user(username, password, email, full_name, role="doctor"):
    """Create a new user account"""
    users = load_users()
    
    # Check if username exists
    if username in users:
        return False, "Username already exists"
    
    # Check if email exists
    for user_data in users.values():
        if user_data.get('email') == email:
            return False, "Email already registered"
    
    # Create new user
    users[username] = {
        'password': hash_password(password),
        'email': email,
        'full_name': full_name,
        'role': role,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_login': None
    }
    
    save_users(users)
    return True, "Account created successfully"

def authenticate_user(username, password):
    """Authenticate a user"""
    users = load_users()
    
    if username not in users:
        return False, "Invalid username or password"
    
    if users[username]['password'] != hash_password(password):
        return False, "Invalid username or password"
    
    # Update last login
    users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_users(users)
    
    return True, users[username]

def change_password(username, old_password, new_password):
    """Change user password"""
    users = load_users()
    
    if username not in users:
        return False, "User not found"
    
    if users[username]['password'] != hash_password(old_password):
        return False, "Current password is incorrect"
    
    users[username]['password'] = hash_password(new_password)
    save_users(users)
    return True, "Password changed successfully"

def logout():
    """Log out the current user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_info = None
    st.rerun()

def authentication_ui():
    """Main authentication interface"""
    
    # Create columns for centering the auth form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1>🦠 NOMA AI</h1>
            <h3>Clinical Skin Analysis System</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Tab selection
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            with st.form("login_form"):
                st.markdown("### Login to Your Account")
                
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                submit = st.form_submit_button("Login", use_container_width=True, type="primary")
                
                if submit:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        success, result = authenticate_user(username, password)
                        
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.session_state.user_info = result
                            st.success(f"Welcome back, {result['full_name']}!")
                            st.rerun()
                        else:
                            st.error(result)
        
        with tab2:
            with st.form("register_form"):
                st.markdown("### Create New Account")
                st.markdown("Fill in the details below to register")
                
                full_name = st.text_input("Full Name*", placeholder="Dr. John Doe")
                email = st.text_input("Email*", placeholder="doctor@hospital.com")
                username = st.text_input("Username*", placeholder="dr_john_doe")
                
                col1, col2 = st.columns(2)
                with col1:
                    password = st.text_input("Password*", type="password", 
                                           placeholder="Min. 8 characters")
                with col2:
                    confirm_password = st.text_input("Confirm Password*", type="password")
                
                role = st.selectbox("Role", ["doctor", "nurse", "researcher", "clinician"])
                
                terms = st.checkbox("I agree to the Terms and Conditions*")
                
                submitted = st.form_submit_button("Register", use_container_width=True, type="primary")
                
                if submitted:
                    # Validation
                    if not all([full_name, email, username, password, confirm_password]):
                        st.error("Please fill in all required fields")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 8:
                        st.error("Password must be at least 8 characters long")
                    elif not terms:
                        st.error("You must agree to the Terms and Conditions")
                    else:
                        # Create account
                        success, message = create_user(username, password, email, full_name, role)
                        
                        if success:
                            st.success("✅ Account created successfully! You can now login.")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")

def user_profile_ui():
    """User profile management interface"""
    with st.expander("👤 User Profile", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            **User:** {st.session_state.user_info.get('full_name', 'N/A')}  
            **Role:** {st.session_state.user_info.get('role', 'N/A')}  
            **Username:** {st.session_state.username}  
            **Email:** {st.session_state.user_info.get('email', 'N/A')}  
            **Member since:** {st.session_state.user_info.get('created_at', 'N/A')}
            """)
        
        with col2:
            # Change password
            with st.form("change_password"):
                st.markdown("#### Change Password")
                old_pw = st.text_input("Current Password", type="password")
                new_pw = st.text_input("New Password", type="password")
                confirm_pw = st.text_input("Confirm New Password", type="password")
                
                if st.form_submit_button("Update Password"):
                    if not old_pw or not new_pw or not confirm_pw:
                        st.error("Please fill in all fields")
                    elif new_pw != confirm_pw:
                        st.error("New passwords do not match")
                    elif len(new_pw) < 8:
                        st.error("Password must be at least 8 characters")
                    else:
                        success, message = change_password(
                            st.session_state.username, 
                            old_pw, 
                            new_pw
                        )
                        if success:
                            st.success("Password updated successfully!")
                        else:
                            st.error(message)

# ==================== LOAD MODEL ====================

@st.cache_resource
def load_ai_model():
    try:
        # Use your specific model name
        model = load_model('noma_cancer_ai_model.keras')
        
        # Get the last convolutional layer for Grad-CAM
        last_conv_layer_name = None
        
        # For MobileNetV3Small, look for the last conv layer in the backbone
        for layer in reversed(model.layers):
            # MobileNetV3Small's convolutional layers are within the functional API
            if 'conv' in layer.name.lower() or 'expand' in layer.name.lower() or 'project' in layer.name.lower():
                if hasattr(layer, 'layers'):  # It's a nested model
                    # Find the deepest conv layer within MobileNetV3Small
                    for sublayer in reversed(layer.layers):
                        if 'conv' in sublayer.name.lower():
                            last_conv_layer_name = sublayer.name
                            break
                elif 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        # Fallback: get the layer before the global average pooling
        if last_conv_layer_name is None:
            for i, layer in enumerate(model.layers):
                if 'global_average_pooling2d' in layer.name.lower():
                    # Get the previous layer
                    last_conv_layer_name = model.layers[i-1].name
                    break
        
        # Another fallback for MobileNetV3
        if last_conv_layer_name is None:
            # MobileNetV3Small typically has its last conv layer named something like this
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if hasattr(sublayer, 'layers'):
                            for deep_layer in sublayer.layers:
                                if 'conv' in deep_layer.name.lower() and 'final' in deep_layer.name.lower():
                                    last_conv_layer_name = deep_layer.name
                                    break
        
        if last_conv_layer_name is None:
            # Final fallback - try to get any conv layer from the MobileNetV3Small
            for layer in model.layers:
                if 'mobilenetv3small' in layer.name.lower() and hasattr(layer, 'layers'):
                    for sublayer in layer.layers:
                        if 'conv' in sublayer.name.lower():
                            last_conv_layer_name = sublayer.name
                            break
            
        st.success(f"✅ Model loaded successfully! Using layer: {last_conv_layer_name}")
        return model, last_conv_layer_name
        
    except Exception as e:
        st.warning(f"Model not found or error loading: {e}. Running in demo mode with simulated predictions.")
        return None, None

model, last_conv_layer = load_ai_model()

# Define classes (update this list to match your model's classes)
classes = [
    "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
    "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
    "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea",
    "Seborrheic Keratoses", "Melanoma",
    "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Sun/Sunlight Damage",
    "Tinea", "Normal", "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
]

malignant_classes = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]
benign_classes = [c for c in classes if c not in malignant_classes + ["Normal"]]
normal_classes = ["Normal"]

# ==================== GRAD-CAM IMPLEMENTATION ====================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    try:
        # Get the last convolutional layer
        last_conv_layer = model.get_layer(last_conv_layer_name)
        
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        # Compute the gradient of the top predicted class for the conv layer output
        with tf.GradientTape() as tape:
            # Convert numpy array to tensor and ensure proper shape
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            conv_outputs, predictions = grad_model(img_tensor)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Gradient of the class output with respect to the feature map
        grads = tape.gradient(class_channel, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)

        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None

def overlay_heatmap(heatmap, image, alpha=0.4):
    """Overlay heatmap on original image"""
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image
            
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        superimposed = cv2.addWeighted(heatmap, alpha, image_np, 1 - alpha, 0)
        return Image.fromarray(superimposed)
    except Exception as e:
        st.error(f"Heatmap overlay error: {e}")
        return image

# ==================== PATIENT MANAGEMENT ====================

def patient_management():
    """Patient registration and selection"""
    st.sidebar.markdown("### 👤 Patient Management")
    
    # Generate unique patient ID
    def generate_patient_id(name, dob):
        unique_string = f"{name}{dob}{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:8]
    
    # Register new patient
    with st.sidebar.expander("➕ Register New Patient", expanded=False):
        with st.form("new_patient"):
            name = st.text_input("Full Name")
            dob = st.date_input("Date of Birth", min_value=datetime(1900,1,1))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            contact = st.text_input("Contact Number")
            email = st.text_input("Email")
            
            if st.form_submit_button("Register Patient"):
                if name and dob:
                    patient_id = generate_patient_id(name, dob)
                    st.session_state.patients[patient_id] = {
                        'id': patient_id,
                        'name': name,
                        'dob': dob.strftime('%Y-%m-%d'),
                        'gender': gender,
                        'contact': contact,
                        'email': email,
                        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'assessments': []
                    }
                    st.session_state.current_patient_id = patient_id
                    st.success(f"Patient registered! ID: {patient_id}")
                    st.rerun()
    
    # Select existing patient
    if st.session_state.patients:
        patient_options = {f"{p['name']} (ID: {pid})": pid 
                          for pid, p in st.session_state.patients.items()}
        selected = st.sidebar.selectbox("Select Patient", list(patient_options.keys()))
        if selected:
            st.session_state.current_patient_id = patient_options[selected]
    
    # Display current patient info
    if st.session_state.current_patient_id:
        patient = st.session_state.patients[st.session_state.current_patient_id]
        st.sidebar.markdown(f"""
        **Current Patient:**  
        👤 {patient['name']}  
        🆔 {patient['id']}  
        📅 Age: {calculate_age(patient['dob'])} years  
        ⚥ {patient['gender']}
        """)

def calculate_age(dob_str):
    """Calculate age from date of birth"""
    dob = datetime.strptime(dob_str, '%Y-%m-%d')
    today = datetime.now()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# ==================== CLINICAL ASSESSMENT WIZARD ====================

def clinical_assessment_wizard():
    """Step-by-step clinical assessment wizard"""
    st.markdown("### 📋 Clinical Assessment Wizard")
    
    # Initialize wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 0
        st.session_state.abcde_answers = {}
        st.session_state.patient_risk = {}
    
    total_steps = 6
    progress = (st.session_state.wizard_step + 1) / total_steps
    st.progress(progress, text=f"Step {st.session_state.wizard_step + 1} of {total_steps}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.wizard_step == 0:
            st.markdown("#### A - ASYMMETRY")
            st.markdown("*Is the lesion asymmetrical?*")
            st.info("Asymmetry means if you draw a line through the middle, the two halves don't match.")
            
            asymmetry = st.radio(
                "Select option:",
                ["No - Symmetrical", "Yes - Asymmetrical"],
                key="asymmetry"
            )
            st.session_state.abcde_answers['asymmetry'] = (asymmetry == "Yes - Asymmetrical")
            
        elif st.session_state.wizard_step == 1:
            st.markdown("#### B - BORDER")
            st.markdown("*Is the border irregular, ragged, or notched?*")
            st.info("Irregular borders look like a coastline with bays, not a smooth, round circle.")
            
            border = st.radio(
                "Select option:",
                ["No - Smooth border", "Yes - Irregular border"],
                key="border"
            )
            st.session_state.abcde_answers['border'] = (border == "Yes - Irregular border")
            
        elif st.session_state.wizard_step == 2:
            st.markdown("#### C - COLOR")
            st.markdown("*How many colors does the lesion have?*")
            st.info("More colors = higher risk. Look for shades of brown, black, red, white, or blue.")
            
            color = st.radio(
                "Select option:",
                ["Single color", "2-3 colors", "Many different colors"],
                key="color"
            )
            st.session_state.abcde_answers['color'] = color.lower().replace(" ", "_")
            
        elif st.session_state.wizard_step == 3:
            st.markdown("#### D - DIAMETER")
            st.markdown("*What is the size of the lesion?*")
            st.info("Measure the widest part. A pencil eraser is about 6mm.")
            
            diameter = st.radio(
                "Select option:",
                ["Small (less than 6mm)", "Medium (6-10mm)", "Large (more than 10mm)"],
                key="diameter"
            )
            st.session_state.abcde_answers['diameter'] = diameter.split()[0].lower()
            
        elif st.session_state.wizard_step == 4:
            st.markdown("#### E - EVOLUTION")
            st.markdown("*Has the lesion changed over time?*")
            st.warning("Recent changes in size, shape, or color are a major warning sign!")
            
            evolution = st.radio(
                "Select option:",
                ["No change - Stable", "Slow change - Months/years", "Rapid change - Weeks/months"],
                key="evolution"
            )
            st.session_state.abcde_answers['evolution'] = evolution.split()[0].lower().replace("-", "").strip()
            
        elif st.session_state.wizard_step == 5:
            st.markdown("#### Patient Risk Factors")
            
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("Age", min_value=1, max_value=120, value=40)
                skin_type = st.selectbox(
                    "Skin Type (Fitzpatrick)",
                    ["I - Always burns, never tans", "II - Usually burns, tans minimally",
                     "III - Sometimes burns, tans gradually", "IV - Rarely burns, tans well",
                     "V - Brown skin, rarely burns", "VI - Black skin, never burns"]
                )
            
            with col_b:
                family_history = st.checkbox("Family history of skin cancer")
                sunburn_history = st.checkbox("History of severe sunburns")
            
            st.session_state.patient_risk = {
                'age': age,
                'skin_type': skin_type[0],  # Just the roman numeral
                'family_history': family_history,
                'sunburn_history': sunburn_history
            }
    
    with col2:
        st.markdown("#### Current Answers")
        if st.session_state.abcde_answers:
            for key, value in st.session_state.abcde_answers.items():
                st.text(f"{key.upper()}: {value}")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.session_state.wizard_step > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.wizard_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.wizard_step < total_steps - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.wizard_step += 1
                st.rerun()
        else:
            if st.button("Calculate Results", use_container_width=True, type="primary"):
                return calculate_risk_score()
    
    return None

def calculate_risk_score():
    """Calculate comprehensive risk score"""
    abcde = st.session_state.abcde_answers
    patient = st.session_state.patient_risk
    
    # ABCDE Score calculation
    abcde_score = 0
    
    # A - Asymmetry
    if abcde.get('asymmetry', False):
        abcde_score += 1
    
    # B - Border
    if abcde.get('border', False):
        abcde_score += 1
    
    # C - Color
    color = abcde.get('color', 'single')
    if color == '2-3_colors':
        abcde_score += 1
    elif color == 'many_different_colors':
        abcde_score += 2
    
    # D - Diameter
    diameter = abcde.get('diameter', 'small')
    if diameter in ['medium', 'large']:
        abcde_score += 1
    
    # E - Evolution
    evolution = abcde.get('evolution', 'no')
    if evolution == 'slow':
        abcde_score += 1
    elif evolution == 'rapid':
        abcde_score += 2
    
    # Patient Risk Score
    patient_score = 0
    if patient.get('age', 40) > 50:
        patient_score += 1
    if patient.get('skin_type', 'III') in ['I', 'II']:
        patient_score += 1
    if patient.get('family_history', False):
        patient_score += 1
    if patient.get('sunburn_history', False):
        patient_score += 1
    
    # Calculate total risk percentage
    total_risk = (abcde_score / 8.0) * 60 + (patient_score / 4.0) * 40
    
    # Determine risk level and recommendations
    if total_risk >= 70:
        risk_level = "HIGH"
        led_color = "RED"
        recommendation = """• URGENT: Dermatology referral (within 1-2 weeks)
• Do not delay evaluation
• Monitor for any changes
• Avoid sun exposure immediately"""
    elif total_risk >= 40:
        risk_level = "MEDIUM"
        led_color = "YELLOW"
        recommendation = """• Schedule dermatology appointment (4-6 weeks)
• Monitor monthly for changes
• Practice sun protection
• Use SPF 30+ daily"""
    else:
        risk_level = "LOW"
        led_color = "GREEN"
        recommendation = """• Continue regular self-checks
• Annual skin examination recommended
• Practice sun safety
• Use SPF 15+ daily"""
    
    return {
        'abcde_score': abcde_score,
        'patient_score': patient_score,
        'total_risk': total_risk,
        'risk_level': risk_level,
        'led_color': led_color,
        'recommendation': recommendation,
        'abcde_details': abcde,
        'patient_details': patient
    }

# ==================== IMAGE ANALYSIS WITH GRAD-CAM ====================

def analyze_image(image, clinical_risk):
    """Analyze skin image with AI model and generate Grad-CAM"""
    
    if model is None:
        # Simulate predictions for demo
        pred_class = np.random.choice(classes)
        confidence = np.random.uniform(0.6, 0.95)
        heatmap_img = None
    else:
        try:
            # Preprocess image for the model
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            
            # Create a properly named input tensor
            # Use dictionary input for the model with the correct layer name
            inputs = {model.input_names[0]: img_array}
            
            # Make prediction using the dictionary input
            predictions = model.predict(inputs, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            pred_class = classes[class_idx] if class_idx < len(classes) else f"Class_{class_idx}"
            
            # Generate Grad-CAM heatmap
            if last_conv_layer:
                # Create a properly formatted input for Grad-CAM
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, class_idx)
                if heatmap is not None:
                    heatmap_img = overlay_heatmap(heatmap, img)
                else:
                    heatmap_img = None
            else:
                heatmap_img = None
                
        except Exception as e:
            st.error(f"Model inference error: {e}")
            pred_class = "Error in analysis"
            confidence = 0.0
            heatmap_img = None
    
    # Determine class type
    if pred_class in malignant_classes:
        class_type = "MALIGNANT"
        base_risk = 80
    elif pred_class in benign_classes:
        class_type = "BENIGN"
        base_risk = 30
    elif pred_class == "Normal":
        class_type = "NORMAL"
        base_risk = 10
    else:
        class_type = "UNKNOWN"
        base_risk = 50
    
    # Combine with clinical risk
    combined_risk = (base_risk * 0.6 + clinical_risk['total_risk'] * 0.4)
    
    return {
        'predicted_class': pred_class,
        'confidence': float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        'class_type': class_type,
        'base_risk': base_risk,
        'combined_risk': combined_risk,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'heatmap': heatmap_img
    }

# ==================== TRACKING DASHBOARD ====================

def tracking_dashboard():
    """Display patient tracking dashboard"""
    st.markdown("### 📊 Longitudinal Tracking Dashboard")
    
    if not st.session_state.current_patient_id:
        st.warning("Please select a patient to view tracking data")
        return
    
    patient = st.session_state.patients[st.session_state.current_patient_id]
    assessments = patient.get('assessments', [])
    
    if not assessments:
        st.info("No assessments yet for this patient")
        return
    
    # Convert assessments to DataFrame
    df = pd.DataFrame(assessments)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Assessments", len(assessments))
    with col2:
        avg_risk = df['combined_risk'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.1f}")
    with col3:
        latest_risk = df['combined_risk'].iloc[-1]
        delta = latest_risk - df['combined_risk'].iloc[0] if len(df) > 1 else 0
        st.metric("Latest Risk", f"{latest_risk:.1f}", f"{delta:+.1f}")
    with col4:
        high_risk_count = len(df[df['combined_risk'] >= 70])
        st.metric("High Risk Alerts", high_risk_count)
    
    # Risk trend chart
    st.markdown("#### Risk Score Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['timestamp']),
        y=df['combined_risk'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10)
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk")
    fig.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
    fig.update_layout(
        title="Risk Score Over Time",
        xaxis_title="Assessment Date",
        yaxis_title="Risk Score",
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Assessment timeline
    st.markdown("#### Assessment Timeline")
    for idx, assessment in enumerate(reversed(assessments[-5:])):
        risk_color = "🔴" if assessment['combined_risk'] >= 70 else "🟡" if assessment['combined_risk'] >= 40 else "🟢"
        
        with st.container():
            st.markdown(f"""
            <div class="timeline-item">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{assessment['timestamp']}</b></span>
                    <span>{risk_color} Risk: {assessment['combined_risk']:.1f}</span>
                </div>
                <div>AI: {assessment['predicted_class']} ({assessment.get('confidence', 0):.1%})</div>
                <div>Type: {assessment['class_type']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Export data
    if st.button("📥 Export Patient Data", use_container_width=True):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"patient_{patient['id']}_data.csv",
            mime="text/csv"
        )

# ==================== MAIN APP ====================

def main():
    # Check if user is authenticated
    if not st.session_state.authenticated:
        authentication_ui()
        return
    
    # If authenticated, show the main app
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🦠 NOMA AI - Clinical Skin Analysis & Tracking</h1>
        <p>Advanced AI-powered skin lesion analysis with Grad-CAM visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add user profile and logout in sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.user_info.get('full_name', 'User')}")
        st.markdown(f"**Role:** {st.session_state.user_info.get('role', 'N/A')}")
        
        # User profile expander
        user_profile_ui()
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        # Patient management
        patient_management()
    
    # Main navigation
    selected = option_menu(
        menu_title=None,
        options=["Clinical Assessment", "Image Analysis", "Tracking Dashboard", "Patient History"],
        icons=["clipboard-check", "camera", "graph-up", "clock-history"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    
    if selected == "Clinical Assessment":
        st.markdown("### 🏥 Step-by-Step Clinical Assessment")
        
        # Run wizard and get results
        risk_results = clinical_assessment_wizard()
        
        if risk_results:
            st.session_state['clinical_risk'] = risk_results
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ABCDE Assessment Results")
                
                # Risk meter
                risk_level = risk_results['risk_level']
                risk_color = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
                
                st.markdown(f"""
                <div class="{risk_color}">
                    <h3 style="text-align: center;">{risk_level} RISK</h3>
                    <p style="text-align: center; font-size: 24px;">{risk_results['total_risk']:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                **ABCDE Score:** {risk_results['abcde_score']}/8  
                **Patient Risk Score:** {risk_results['patient_score']}/4  
                **LED Indicator:** {risk_results['led_color']}
                """)
            
            with col2:
                st.markdown("#### Clinical Recommendations")
                st.info(risk_results['recommendation'])
            
            # Store in session
            st.session_state['clinical_complete'] = True
            
            if st.button("Proceed to Image Analysis →", use_container_width=True):
                st.session_state['nav_to_image'] = True
                st.rerun()
    
    elif selected == "Image Analysis":
        st.markdown("### 📸 Skin Image Analysis with Grad-CAM")
        
        if 'clinical_risk' not in st.session_state:
            st.warning("Please complete clinical assessment first")
            if st.button("Go to Clinical Assessment", use_container_width=True):
                st.session_state['nav_to_clinical'] = True
                st.rerun()
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Upload or Capture Image")
                
                # Image input options
                img_source = st.radio("Image Source:", ["Upload", "Camera"])
                
                if img_source == "Upload":
                    uploaded_file = st.file_uploader(
                        "Choose an image...", 
                        type=['jpg', 'jpeg', 'png'],
                        help="Upload a clear image of the skin lesion"
                    )
                    if uploaded_file:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                else:
                    camera_image = st.camera_input("Take a picture")
                    if camera_image:
                        image = Image.open(camera_image)
                
                if 'image' in locals():
                    if st.button("🔬 Analyze Image with Grad-CAM", use_container_width=True, type="primary"):
                        with st.spinner("Analyzing image with AI and generating Grad-CAM..."):
                            # Analyze image with Grad-CAM
                            ai_results = analyze_image(image, st.session_state['clinical_risk'])
                            st.session_state['ai_results'] = ai_results
                            
                            # Save assessment
                            if st.session_state.current_patient_id:
                                patient = st.session_state.patients[st.session_state.current_patient_id]
                                assessment = {
                                    **ai_results,
                                    'clinical_risk': st.session_state['clinical_risk']['total_risk'],
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }
                                # Remove heatmap from saved data (can't serialize image)
                                if 'heatmap' in assessment:
                                    del assessment['heatmap']
                                patient['assessments'].append(assessment)
                            
                            st.success("Analysis complete!")
                            st.rerun()
            
            with col2:
                if 'ai_results' in st.session_state:
                    results = st.session_state['ai_results']
                    clinical = st.session_state['clinical_risk']
                    
                    st.markdown("#### Analysis Results")
                    
                    # Risk display
                    combined_risk = results['combined_risk']
                    if combined_risk >= 70:
                        st.markdown(f"""
                        <div class="risk-high">
                            <h4 style="text-align: center;">HIGH RISK</h4>
                            <p style="text-align: center; font-size: 20px;">{combined_risk:.1f}/100</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif combined_risk >= 40:
                        st.markdown(f"""
                        <div class="risk-medium">
                            <h4 style="text-align: center;">MEDIUM RISK</h4>
                            <p style="text-align: center; font-size: 20px;">{combined_risk:.1f}/100</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h4 style="text-align: center;">LOW RISK</h4>
                            <p style="text-align: center; font-size: 20px;">{combined_risk:.1f}/100</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    **AI Prediction:** {results['predicted_class']}  
                    **Confidence:** {results['confidence']:.1%}  
                    **Lesion Type:** {results['class_type']}  
                    **Clinical Risk:** {clinical['total_risk']:.1f}/100  
                    **Combined Score:** {results['combined_risk']:.1f}/100
                    """)
                    
                    # Display Grad-CAM heatmap
                    if results.get('heatmap') is not None:
                        st.markdown("#### 🔥 Grad-CAM Visualization")
                        st.markdown("""
                        <div class="heatmap-container">
                            <p style="text-align: center;"><i>Areas in red show regions the AI focused on for its decision</i></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display original and heatmap side by side
                        heat_col1, heat_col2 = st.columns(2)
                        with heat_col1:
                            st.image(image, caption="Original Image", use_container_width=True)
                        with heat_col2:
                            st.image(results['heatmap'], caption="Grad-CAM Heatmap", use_container_width=True)
                    
                    # Action buttons
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📋 Save to Record", use_container_width=True):
                            st.success("Assessment saved to patient record!")
                    with col_b:
                        if st.button("🔄 New Assessment", use_container_width=True):
                            for key in ['clinical_risk', 'ai_results', 'clinical_complete']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
    
    elif selected == "Tracking Dashboard":
        tracking_dashboard()
    
    elif selected == "Patient History":
        st.markdown("### 📜 Complete Patient History")
        
        if not st.session_state.current_patient_id:
            st.warning("Please select a patient")
            return
        
        patient = st.session_state.patients[st.session_state.current_patient_id]
        assessments = patient.get('assessments', [])
        
        if assessments:
            # Summary statistics
            df = pd.DataFrame(assessments)
            
            # Advanced analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Risk Distribution")
                fig = px.pie(
                    names=['Low Risk (<40)', 'Medium Risk (40-70)', 'High Risk (>70)'],
                    values=[
                        len(df[df['combined_risk'] < 40]),
                        len(df[(df['combined_risk'] >= 40) & (df['combined_risk'] < 70)]),
                        len(df[df['combined_risk'] >= 70])
                    ],
                    color_discrete_sequence=['#26de81', '#ffa502', '#ff4757']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Condition Frequency")
                top_conditions = df['predicted_class'].value_counts().head(5)
                if not top_conditions.empty:
                    fig = px.bar(
                        x=top_conditions.values,
                        y=top_conditions.index,
                        orientation='h',
                        title="Most Common Conditions"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Full history table
            st.markdown("#### All Assessments")
            display_df = df[['timestamp', 'predicted_class', 'class_type', 'confidence', 'combined_risk']]
            display_df.columns = ['Date', 'Condition', 'Type', 'Confidence', 'Risk Score']
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            display_df['Risk Score'] = display_df['Risk Score'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No assessment history available")

if __name__ == "__main__":
    main()
