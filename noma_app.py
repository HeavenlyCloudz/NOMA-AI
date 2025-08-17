import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QApplication, QFileDialog, QTextEdit, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image
import platform
from datetime import datetime
import os

# Conditional GPIO import
if platform.system() == "Linux" and "arm" in platform.uname().machine:
    from gpiozero import LED
else:
    class LED:
        def __init__(self, pin):
            self.pin = pin

        def on(self):
            print(f"LED on pin {self.pin} is turned ON (mock).")

        def off(self):
            print(f"LED on pin {self.pin} is turned OFF (mock).")

class ModelLoader(QThread):
    model_loaded = pyqtSignal()
    load_failed = pyqtSignal(str)

    def run(self):
        try:
            self.model = load_model('right_noma_model.keras')
            self.model_loaded.emit()
        except Exception as e:
            self.load_failed.emit(str(e))

class NomaAIApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # GPIO setup
        self.red_light = LED(17)
        self.yellow_light = LED(27)
        self.green_light = LED(22)

        # Classification categories
        self.classes = [
            "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
            "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
            "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea",
            "Seborrheic Keratoses", "Melanoma",  # Malignant
            "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Sun/Sunlight Damage",
            "Tinea", "Normal", "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
        ]

        self.malignant_classes = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]
        self.benign_classes = [
            "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous", "Candidiasis",
            "Drug Eruption", "Eczema", "Infestations/Bites", "Lichen", "Lupus",
            "Moles", "Psoriasis", "Rosacea", "Seborrheic Keratoses",
            "Sun/Sunlight Damage", "Tinea",
            "Vascular Tumors", "Vasculitis", "Vitiligo", "Warts"
        ]
        self.normal_classes = ["Normal"]

        self.initUI()
        self.load_model()

    def initUI(self):
        self.setWindowTitle("NOMA AI Skin Cancer Detection")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.image_label = QLabel("No image uploaded")
        self.image_label.setAlignment(QtGui.Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        self.classify_button = QPushButton("Classify")
        self.classify_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)  # Disabled until model loads
        layout.addWidget(self.classify_button)

        self.feedback_area = QTextEdit("Provide feedback...")
        layout.addWidget(self.feedback_area)

        self.submit_feedback_button = QPushButton("Submit Feedback")
        self.submit_feedback_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.submit_feedback_button.clicked.connect(self.submit_feedback)
        layout.addWidget(self.submit_feedback_button)

        # Shutdown button
        self.shutdown_button = QPushButton("Shutdown Device")
        self.shutdown_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.shutdown_button.clicked.connect(self.shutdown_device)
        layout.addWidget(self.shutdown_button)

        self.setLayout(layout)

    def load_model(self):
        self.loading_label = QLabel("Loading model, please wait...")
        self.layout().addWidget(self.loading_label)

        self.model_loader = ModelLoader()
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.load_failed.connect(self.on_model_load_failed)
        self.model_loader.start()

    def on_model_loaded(self):
        self.loading_label.setText("Model loaded successfully!")
        self.loading_label.setStyleSheet("color: green;")
        self.classify_button.setEnabled(True)

    def on_model_load_failed(self, error_message):
        self.loading_label.setText(f"Failed to load model: {error_message}")
        self.loading_label.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Error", f"Model loading failed:\n{error_message}")

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_label.setPixmap(QtGui.QPixmap(file_name).scaled(400, 300))
            self.image_path = file_name

    def classify_image(self):
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return

        try:
            image = Image.open(self.image_path).convert("RGB")
            img_array = self.preprocess_image(image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {e}")
            return

        try:
            predictions = self.model_loader.model.predict(img_array)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {e}")
            return

        class_index = np.argmax(predictions[0])
        predicted_class = self.classes[class_index]

        # LED control
        if predicted_class in self.malignant_classes:
            result = "Malignant"
            self.red_light.on()
            self.yellow_light.off()
            self.green_light.off()
        elif predicted_class in self.benign_classes:
            result = "Benign"
            self.red_light.off()
            self.yellow_light.on()
            self.green_light.off()
        elif predicted_class in self.normal_classes:
            result = "Normal"
            self.red_light.off()
            self.yellow_light.off()
            self.green_light.on()

        QMessageBox.information(self, "Prediction Result", f"Predicted Class: {predicted_class}\nClassification: {result}")

    def preprocess_image(self, image):
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def submit_feedback(self):
        feedback = self.feedback_area.toPlainText().strip()
        if feedback:
            with open("user_feedback.txt", "a") as f:
                f.write(f"{datetime.now()} - Feedback: {feedback}\n{'-'*50}\n")
            QMessageBox.information(self, "Feedback", "Thank you for your feedback!")
        else:
            QMessageBox.warning(self, "Warning", "Please enter feedback before submitting.")

    def shutdown_device(self):
        reply = QMessageBox.question(
            self, 'Shutdown Confirmation', 'Are you sure you want to turn off the device?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.system("sudo shutdown now")  # Safe shutdown for Raspberry Pi

    def closeEvent(self, event):
        self.red_light.off()  # Reset lights
        self.yellow_light.off()
        self.green_light.off()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NomaAIApp()
    ex.show()
    sys.exit(app.exec_())
