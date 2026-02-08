# **NOMA AI — The Visual-Clinic Integrator**

NOMA AI is a portable, AI-powered skin cancer screening system that fuses computer vision with clinical risk assessment to provide accessible, preliminary skin lesion evaluation. Built on a Raspberry Pi 4, it democratizes dermatological screening through an integrated hardware–software solution designed for point-of-care use.

# Hardware Overview

- Raspberry Pi 4 (4 GB RAM): Real-time AI inference and system control

- Arducam 12 MP IMX519: High-resolution autofocus imaging for lesion detail capture

- Waveshare 5” Touchscreen: Live camera feed and interactive GUI

- Tri-Color LED System (Red/Yellow/Green): Immediate physical risk indication

- Portable Enclosure & Powerbank: Fully self-contained, mobile device

# Dataset & Preprocessing

- Dataset: 12,900 dermoscopic and standard images across 24 skin conditions

- Classes: 4 malignant, 20 benign

- Split: 80% training / 20% validation

- Input Size: 224×224, normalized (0–1)

- Augmentation: Flip, rotation, zoom, contrast, brightness

- Class Imbalance: Handled using computed class weights

# Model Architecture

- Base Model: MobileNetV3 (ImageNet pre-trained)

- Framework: TensorFlow 2.18

- Output: 24-class softmax classifier

# Deployment

Formats: .keras and optimized .tflite

Optimization: tf.lite.Optimize.DEFAULT

Inference: Runs fully on-device (Raspberry Pi)

**# NOMA AI App & Risk Fusion**

NOMA AI does not rely on AI alone. Each scan combines:

- AI visual classification

- ABCDE clinical risk assessment (Asymmetry, Border, Color, Diameter, Evolution)

- Grad-CAM heatmaps for model interpretability

# LED Alerts:

🔴 Red — High risk (seek medical attention)

🟡 Yellow — Monitor lesion

🟢 Green — Low risk / normal

# Purpose

NOMA AI is designed as an early-warning and educational tool, not a diagnostic replacement. By integrating explainable AI with clinician-inspired reasoning, it strengthens the first line of defense against skin cancer while improving public health literacy.

_Make startup script executable_

chmod +x /home/havil/noma_ai/start_noma.sh

_Install service_

sudo cp noma_ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable noma_ai.service
sudo systemctl start noma_ai.service

_Check status_

sudo systemctl status noma_ai.service

_View logs_

sudo journalctl -u noma_ai.service -f
