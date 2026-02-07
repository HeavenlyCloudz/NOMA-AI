 NOMA AI: AI-Powered Skin Cancer Detection System





NOMA AI is a comprehensive, accessible skin cancer screening system that combines artificial intelligence with clinical decision support. Built for the Raspberry Pi platform, it provides an all-in-one solution for preliminary skin lesion analysis—making early detection more accessible to everyone.

🎯 Project Vision
Skin cancer is the most common cancer worldwide, with early detection dramatically improving survival rates. NOMA AI addresses this challenge by creating an affordable, user-friendly screening tool that combines:


AI-powered visual analysis using a custom-trained CNN


Clinical risk assessment following established ABCDE guidelines


Physical LED triage system for clear, immediate feedback


Educational interface that teaches users about skin health



✨ Key Features
🤖 AI Analysis


24-class CNN model trained on dermatological images


Grad-CAM visualization showing what the AI sees


Real-time image capture with Arducam IMX519 (16MP)


Confidence-based recommendations


🏥 Clinical Integration


Interactive ABCDE assessment (Asymmetry, Border, Color, Diameter, Evolution)


Patient risk profiling (age, skin type, family history)


Step-by-step wizard with progress tracking


Combined AI + clinical risk scoring


💡 Hardware Integration


Three-color LED system (Red / Yellow / Green) for immediate triage


Fullscreen touch interface optimized for Raspberry Pi 7" display


EGLFS display backend for smooth kiosk operation


GPIO-controlled physical indicators


📊 Results & Reporting


Clear risk categorization (High / Moderate / Low)


Actionable recommendations with next steps


Visual heatmaps highlighting areas of concern


Educational explanations of findings



🛠️ Technical Stack
ComponentTechnologyHardwareRaspberry Pi 4, Arducam IMX519, 3× LEDs, 7" Touch DisplayAI FrameworkTensorFlow Lite (TFLite)GUI FrameworkPyQt5 with EGLFS backendCameraPicamera2Image ProcessingOpenCV, PILVisualizationGrad-CAM, MatplotlibGPIO ControlGPIOZero

🚀 Getting Started
Prerequisites


Raspberry Pi 4 (4GB+ recommended)


Arducam IMX519 camera module


7" Raspberry Pi touch display


3× LEDs (Red, Yellow, Green) with resistors


Raspberry Pi OS (Bookworm, 64-bit)



Installation
# 1. Clone the repository
git clone https://github.com/yourusername/noma-ai.git
cd noma-ai

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Arducam drivers (IMX519)
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps

# 5. Enable kiosk mode
sudo cp noma_ai.service /etc/systemd/system/
sudo systemctl enable noma_ai.service
sudo systemctl start noma_ai.service


🔌 Hardware Setup
LED Connections:
- Red LED:    GPIO 17
- Yellow LED: GPIO 27
- Green LED:  GPIO 22

Camera:  CSI port
Display: DSI port


📖 User Guide
Basic Workflow


Power on the device — system boots directly into NOMA AI


Position lesion in camera view (ensure good lighting)


Click START ANALYSIS — image captured and AI runs


Complete clinical questions (ABCDE assessment)


Review results — risk level, heatmap, and recommendations


Follow LED guidance


LED Indicators
LED StateMeaningAction🔴 Solid RedHigh RiskUrgent dermatology referral🟡 Solid YellowModerate RiskAppointment within 4–6 weeks🟢 Solid GreenLow RiskContinue regular self-checks🟡 Blinking YellowAssessment ActiveAnswering questions🟢 Green PatternAssessment CompleteResults ready

🧠 AI Model Details


Architecture: Custom CNN


Classes: 24 dermatological conditions


4 malignant


19 benign


1 normal




Input: 224 × 224 RGB images


Output: Probability distribution across classes


Training Data: Curated dermatology image dataset


Accuracy: >85% validation accuracy (medical validation pending)



🏗️ Project Structure
noma-ai/
├── noma_app.py              # Main application
├── requirements.txt         # Python dependencies
├── launch_eglfs.sh          # Startup script
├── noma_ai.service          # Systemd service
├── models/
│   └── nomaai_model.tflite  # AI model
├── docs/
│   └── hardware_setup.md    # Hardware guide
└── tests/
    └── test_components.py   # Tests


⚠️ Important Disclaimer
NOMA AI IS NOT A MEDICAL DEVICE AND DOES NOT PROVIDE MEDICAL DIAGNOSIS.
This system is intended for:


Educational purposes


Preliminary screening and awareness


Complementing professional medical care


Always consult a qualified dermatologist or healthcare professional for diagnosis and treatment.

🔬 Research & Development
NOMA AI contributes to research in:


Accessible medical AI systems


Human–AI collaboration in healthcare


Edge computing for medical applications


Explainable AI in clinical contexts



🤝 Contributing
Contributions are welcome!
Areas of interest include:


Model training and improvement


UI/UX enhancements


Multilingual support


Hardware optimization


Clinical validation studies


See CONTRIBUTING.md for details.

📄 License
This project is licensed under the MIT License.
See the LICENSE file for more information.

📚 Citations & References
If you use NOMA AI in research, please cite:
@software{noma_ai_2024,
  title  = {NOMA AI: AI-Powered Skin Cancer Detection System},
  author = {Havil},
  year   = {2026},
  url    = {https://github.com/yourusername/noma-ai}
}
