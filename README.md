NOMA AI
AI-Powered Skin Cancer Detection System

NOMA AI is an accessible skin cancer screening system that combines artificial intelligence with clinical decision support. Built for the Raspberry Pi, it provides a low-cost, edge-based solution for preliminary skin lesion analysis and skin health education.

Project Vision

Skin cancer is the most common cancer worldwide, and early detection significantly improves outcomes. NOMA AI aims to increase accessibility to early screening by combining:

AI-based visual analysis using a custom CNN

Clinical risk assessment based on ABCDE guidelines

Physical LED triage for immediate feedback

An educational interface focused on skin health awareness

Key Features
AI Analysis

Custom 24-class convolutional neural network

Grad-CAM explainability to visualize model attention

Real-time image capture using a 16MP Arducam IMX519

Confidence-aware predictions and recommendations

Clinical Decision Support

Interactive ABCDE assessment workflow

Patient risk profiling (age, skin type, family history)

Step-by-step guided interface with progress tracking

Combined AI output and clinical risk scoring

Hardware Integration

Three-color LED triage system (red, yellow, green)

Fullscreen touch interface optimized for Raspberry Pi

EGLFS backend for kiosk-style deployment

GPIO-controlled physical indicators

Results and Feedback

Clear risk categorization (high, moderate, low)

Actionable next-step recommendations

Visual heatmaps highlighting areas of concern

Plain-language explanations of findings

Technology Stack

Raspberry Pi 4

Arducam IMX519 camera

TensorFlow Lite

PyQt5 (EGLFS backend)

OpenCV and PIL

GPIOZero

Grad-CAM for explainability

Getting Started
Requirements

Raspberry Pi 4 (4GB+ recommended)

Arducam IMX519 camera module

7-inch Raspberry Pi touch display

Three LEDs with resistors

Raspberry Pi OS (Bookworm, 64-bit)

Installation
git clone https://github.com/yourusername/noma-ai.git
cd noma-ai

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps

sudo cp noma_ai.service /etc/systemd/system/
sudo systemctl enable noma_ai.service
sudo systemctl start noma_ai.service

Hardware Setup

Red LED: GPIO 17

Yellow LED: GPIO 27

Green LED: GPIO 22

Camera: CSI port

Display: DSI port

User Workflow

Power on the device and launch NOMA AI

Position the lesion in the camera view

Start AI analysis

Complete the ABCDE clinical assessment

Review risk level, heatmap, and recommendations

Follow LED guidance

AI Model Overview

Custom convolutional neural network

24 dermatological classes

224 × 224 RGB input images

Probability-based multi-class output

Validation accuracy above 85 percent

Medical validation pending

Project Structure
noma-ai/
├── noma_app.py
├── requirements.txt
├── launch_eglfs.sh
├── noma_ai.service
├── models/
│   └── nomaai_model.tflite
├── docs/
│   └── hardware_setup.md
└── tests/
    └── test_components.py

Disclaimer

NOMA AI is not a medical device and does not provide medical diagnosis.

This system is intended for educational use, preliminary screening, and awareness building only. It is not a replacement for professional medical evaluation.

Always consult a qualified healthcare professional for diagnosis and treatment.

Research Focus

NOMA AI supports research in:

Accessible medical AI

Edge computing in healthcare

Human-AI collaboration

Explainable AI for clinical use

Contributing

Contributions are welcome, including:

Model improvement

Interface enhancements

Hardware optimization

Clinical validation research

License

This project is licensed under the MIT License.

Citation
@software{noma_ai_2024,
  title  = {NOMA AI: AI-Powered Skin Cancer Detection System},
  author = {Havil},
  year   = {2024},
  url    = {https://github.com/yourusername/noma-ai}
}
