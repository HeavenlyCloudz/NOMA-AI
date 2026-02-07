🏥 NOMA AI: AI-Powered Skin Cancer Detection System
https://img.shields.io/badge/Platform-Raspberry%2520Pi%25204-FF6F61
https://img.shields.io/badge/Python-3.9%252B-blue
https://img.shields.io/badge/GUI-PyQt5-green
https://img.shields.io/badge/AI-TensorFlow%2520Lite-orange
https://img.shields.io/badge/License-MIT-yellow

NOMA AI is a comprehensive, accessible skin cancer screening system that combines artificial intelligence with clinical decision support. Built for the Raspberry Pi platform, it provides an all-in-one solution for preliminary skin lesion analysis, making early detection more accessible to everyone.

🎯 Project Vision
Skin cancer is the most common cancer worldwide, with early detection dramatically improving survival rates. NOMA AI addresses this by creating an affordable, user-friendly screening tool that combines:

AI-powered visual analysis using a custom-trained CNN

Clinical risk assessment following established ABCDE guidelines

Physical LED triage system for clear, immediate feedback

Educational interface that teaches users about skin health

✨ Key Features
🤖 AI Analysis
24-class CNN model trained on dermatological images

Grad-CAM visualization showing "what the AI sees"

Real-time image capture with Arducam IMX519 (16MP)

Confidence-based recommendations

🏥 Clinical Integration
Interactive ABCDE assessment (Asymmetry, Border, Color, Diameter, Evolution)

Patient risk profiling (age, skin type, family history)

Step-by-step wizard with progress tracking

Combined AI + clinical risk scoring

💡 Hardware Integration
Three-color LED system (Red/Yellow/Green) for immediate triage

Fullscreen touch interface optimized for Raspberry Pi 7" display

EGLFS display backend for smooth kiosk operation

GPIO-controlled physical indicators

📊 Results & Reporting
Clear risk categorization (High/Moderate/Low)

Actionable recommendations with next steps

Visual heatmaps highlighting areas of concern

Educational explanations of findings

🛠️ Technical Stack
Component	Technology
Hardware	Raspberry Pi 4, Arducam IMX519, 3x LEDs, 7" Touch Display
AI Framework	TensorFlow Lite (TFLite)
GUI Framework	PyQt5 with EGLFS backend
Camera	Picamera2 library
Image Processing	OpenCV, PIL
Visualization	Grad-CAM, Matplotlib
GPIO Control	GPIOZero
🚀 Getting Started
Prerequisites
Raspberry Pi 4 (4GB+ recommended)

Arducam IMX519 camera module

7" Raspberry Pi touch display

3x LEDs (Red, Yellow, Green) with resistors

Raspberry Pi OS (Bookworm 64-bit)

Installation
bash
# 1. Clone repository
git clone https://github.com/yourusername/noma-ai.git
cd noma-ai

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Arducam drivers (if using IMX519)
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps

# 5. Configure for kiosk mode
sudo cp noma_ai.service /etc/systemd/system/
sudo systemctl enable noma_ai.service
sudo systemctl start noma_ai.service
Hardware Setup
text
LED Connections:
- Red LED: GPIO 17
- Yellow LED: GPIO 27  
- Green LED: GPIO 22

Camera: CSI port
Display: DSI port
📖 User Guide
Basic Workflow
Power on the device - System boots directly to NOMA AI interface

Position lesion in camera view - Ensure good lighting and focus

Click "START ANALYSIS" - Captures image and begins AI analysis

Answer clinical questions - Step-by-step ABCDE assessment

Review results - See risk level, recommendations, and heatmap

Follow LED guidance - Red/Yellow/Green indicates risk level

LED Indicators
LED State	Meaning	Action
🔴 Solid Red	High Risk	Urgent dermatology referral
🟡 Solid Yellow	Moderate Risk	Schedule appointment within 4-6 weeks
🟢 Solid Green	Low Risk	Continue regular self-checks
🟡 Blinking Yellow	Assessment Active	Answering clinical questions
🟢 Pattern Green	Assessment Complete	Results ready for review
🧠 AI Model Details
Architecture: Custom convolutional neural network

Classes: 24 dermatological conditions (4 malignant, 19 benign, 1 normal)

Input: 224×224 RGB images

Output: Probability distribution across classes

Training Data: Curated dataset of dermatological images

Accuracy: >85% on validation set (medical validation pending)

🏗️ Project Structure
text
noma-ai/
├── noma_app.py              # Main application
├── requirements.txt         # Python dependencies
├── launch_eglfs.sh         # Startup script
├── noma_ai.service         # Systemd service file
├── models/
│   └── nomaai_model.tflite # AI model
├── docs/
│   └── hardware_setup.md   # Hardware assembly guide
└── tests/
    └── test_components.py  # Hardware/software tests
⚠️ Important Disclaimer
NOMA AI IS NOT A MEDICAL DEVICE AND DOES NOT PROVIDE MEDICAL DIAGNOSIS.

This system is designed for:

Educational purposes to learn about skin health

Preliminary screening to identify potential concerns

Awareness building about skin cancer detection

Complementing professional medical care

ALWAYS CONSULT A QUALIFIED DERMATOLOGIST OR HEALTHCARE PROFESSIONAL FOR PROPER DIAGNOSIS AND TREATMENT.

🔬 Research & Development
NOMA AI represents ongoing research in:

Accessible medical AI systems

Human-AI collaboration in clinical decision making

Edge computing for healthcare applications

Explainable AI in medical contexts

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Areas for contribution:

Model improvement and training

UI/UX enhancements

Additional language support

Hardware optimizations

Clinical validation studies

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

📚 Citations & References
If you use NOMA AI in research, please cite:

bibtex
@software{noma_ai_2024,
  title = {NOMA AI: AI-Powered Skin Cancer Detection System},
  author = {Havil},
  year = {2024},
  url = {https://github.com/yourusername/noma-ai}
}
