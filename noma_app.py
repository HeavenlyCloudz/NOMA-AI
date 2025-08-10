import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QPushButton, QApplication, QFileDialog, QTextEdit, QMessageBox
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
import platform

# Conditional GPIO import
if platform.system() == "Linux" and "arm" in platform.uname().machine:
    from gpiozero import LED
else:
    class LED:
        def __init__(self, pin):
            self.pin = pin  # Store pin number for reference

        def on(self):
            print(f"LED on pin {self.pin} is turned ON (mock).")  # Mock behavior

        def off(self):
            print(f"LED on pin {self.pin} is turned OFF (mock).")  # Mock behavior

class ModelLoader(QThread):
    model_loaded = pyqtSignal()

    def run(self):
        self.model = load_model('right_noma_model.keras')  # Updated model name
        self.model_loaded.emit()

class NomaAIApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Set up GPIO using GPIOZero conditionally
        self.red_light = LED(17)    # GPIO pin for red light
        self.yellow_light = LED(27)  # GPIO pin for yellow light
        self.green_light = LED(22)   # GPIO pin for green light

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
        layout.addWidget(self.classify_button)

        self.feedback_area = QTextEdit("Provide feedback...")
        layout.addWidget(self.feedback_area)

        self.submit_feedback_button = QPushButton("Submit Feedback")
        self.submit_feedback_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.submit_feedback_button.clicked.connect(self.submit_feedback)
        layout.addWidget(self.submit_feedback_button)

        self.setLayout(layout)

    def load_model(self):
        self.loading_label = QLabel("Loading model, please wait...")
        self.layout().addWidget(self.loading_label)

        self.model_loader = ModelLoader()
        self.model_loader.model_loaded.connect(self.on_model_loaded)
        self.model_loader.start()

    def on_model_loaded(self):
        self.loading_label.setText("Model loaded successfully!")
        self.loading_label.setStyleSheet("color: green;")

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

        image = Image.open(self.image_path)
        img_array = self.preprocess_image(image)

        # Ensure the model is loaded before predicting
        if not hasattr(self, 'model_loader') or not hasattr(self.model_loader, 'model'):
            QMessageBox.warning(self, "Warning", "Model is not loaded yet.")
            return

        predictions = self.model_loader.model.predict(img_array)
        class_index = np.argmax(predictions[0])
        predicted_class = self.classes[class_index]

        # Control lights based on classification
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
        feedback = self.feedback_area.toPlainText()
        if feedback:
            with open("user_feedback.txt", "a") as f:
                f.write(f"Feedback: {feedback}\n{'-'*50}\n")
            QMessageBox.information(self, "Feedback", "Thank you for your feedback!")
        else:
            QMessageBox.warning(self, "Warning", "Please enter feedback before submitting.")

    def closeEvent(self, event):
        self.red_light.off()  # Reset lights
        self.yellow_light.off()
        self.green_light.off()
        event.accept()  # Accept event to close the app

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NomaAIApp()
    ex.show()
    sys.exit(app.exec_())
