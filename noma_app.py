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
import cv2

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

# ---------------- Camera Thread ---------------- #
class CameraThread(QThread):
    frame_ready = pyqtSignal(QtGui.QImage)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self.running:
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# ---------------- Model Loader ---------------- #
class ModelLoader(QThread):
    model_loaded = pyqtSignal()
    load_failed = pyqtSignal(str)

    def run(self):
        try:
            self.model = load_model('right_noma_model.keras')
            self.model_loaded.emit()
        except Exception as e:
            self.load_failed.emit(str(e))

# ---------------- Main App ---------------- #
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
        self.start_camera()

    def initUI(self):
        self.setWindowTitle("NOMA AI Skin Cancer Detection")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        # Image / camera display
        self.image_label = QLabel("Loading camera feed...")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Buttons
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        self.classify_button = QPushButton("Classify")
        self.classify_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)
        layout.addWidget(self.classify_button)

        self.feedback_area = QTextEdit("Provide feedback...")
        layout.addWidget(self.feedback_area)

        self.submit_feedback_button = QPushButton("Submit Feedback")
        self.submit_feedback_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.submit_feedback_button.clicked.connect(self.submit_feedback)
        layout.addWidget(self.submit_feedback_button)

        self.shutdown_button = QPushButton("Shutdown Device")
        self.shutdown_button.setStyleSheet("font-size: 24px; padding: 20px;")
        self.shutdown_button.clicked.connect(self.shutdown_device)
        layout.addWidget(self.shutdown_button)

        self.setLayout(layout)

    # ---------------- Camera ---------------- #
    def start_camera(self):
        self.camera_thread = CameraThread(camera_index=0)
        self.camera_thread.frame_ready.connect(self.update_camera_feed)
        self.camera_thread.start()

    def update_camera_feed(self, qt_image):
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qt_image).scaled(400, 300))
        self.current_frame = qt_image

    # ---------------- Model ---------------- #
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

    # ---------------- Image Upload ---------------- #
    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_label.setPixmap(QtGui.QPixmap(file_name).scaled(400, 300))
            self.image_path = file_name

    # ---------------- Classification ---------------- #
    def classify_image(self):
        if hasattr(self, 'image_path'):
            image = Image.open(self.image_path).convert("RGB")
        elif hasattr(self, 'current_frame'):
            image = Image.frombytes(
                "RGB",
                (self.current_frame.width(), self.current_frame.height()),
                self.current_frame.bits().asstring(self.current_frame.byteCount())
            )
        else:
            QMessageBox.warning(self, "Warning", "No image available.")
            return

        img_array = self.preprocess_image(image)
        predictions = self.model_loader.model.predict(img_array)
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

    # ---------------- Feedback ---------------- #
    def submit_feedback(self):
        feedback = self.feedback_area.toPlainText().strip()
        if feedback:
            with open("user_feedback.txt", "a") as f:
                f.write(f"{datetime.now()} - Feedback: {feedback}\n{'-'*50}\n")
            QMessageBox.information(self, "Feedback", "Thank you for your feedback!")
        else:
            QMessageBox.warning(self, "Warning", "Please enter feedback before submitting.")

    # ---------------- Shutdown ---------------- #
    def shutdown_device(self):
        reply = QMessageBox.question(
            self, 'Shutdown Confirmation', 'Are you sure you want to turn off the device?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            os.system("sudo shutdown now")

    # ---------------- Close Event ---------------- #
    def closeEvent(self, event):
        self.camera_thread.stop()
        self.red_light.off()
        self.yellow_light.off()
        self.green_light.off()
        event.accept()

# ---------------- Main ---------------- #
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NomaAIApp()
    ex.show()
    sys.exit(app.exec_())
