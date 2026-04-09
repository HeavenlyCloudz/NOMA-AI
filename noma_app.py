#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Force Qt to use eglfs and clean up environment
import os
os.environ['QT_QPA_PLATFORM'] = 'eglfs'
os.environ['QT_QPA_EGLFS_HIDECURSOR'] = '1'
os.environ['QT_QPA_EGLFS_WIDTH'] = '800'
os.environ['QT_QPA_EGLFS_HEIGHT'] = '480'

# Remove conflicting plugin paths
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
if 'QT_PLUGIN_PATH' in os.environ:
    del os.environ['QT_PLUGIN_PATH']

# Camera settings
os.environ['LIBCAMERA_LOG_LEVELS'] = '0'
os.environ['LIBCAMERA_IPA'] = 'rpi/vc4'

import sys
import time
import json
import random
import numpy as np
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QPushButton, QApplication,
                             QMessageBox, QProgressBar, QMainWindow, QScrollArea,
                             QWidget, QHBoxLayout, QTextEdit, QRadioButton,
                             QSpinBox, QComboBox, QCheckBox, QGroupBox, QTabWidget)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
from PIL import Image
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

# ---------------- Camera Permission Fix ---------------- #
os.environ['LIBCAMERA_LOG_LEVELS'] = '0'
os.environ['LIBCAMERA_IPA'] = 'rpi/vc4'

# ---------------- Global Exception Handler ---------------- #
def exception_handler(exc_type, exc_value, exc_traceback):
    print("=== CRASH DETECTED ===")
    print(f"Exception: {exc_type.__name__}: {exc_value}")
    with open('/home/havil/noma_ai/crash_log.txt', 'a') as f:
        f.write(f"Crash at: {datetime.now()}\n")
        f.write(f"Exception: {exc_type.__name__}: {exc_value}\n")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_handler
print("Global exception handler installed")

# ---------------- DISEASE INFORMATION DATABASE ---------------- #
DISEASE_INFO = {
    "Melanoma": {
        "description": "Most serious form of skin cancer. Develops in melanocytes (pigment-producing cells). Can spread quickly if not caught early.",
        "warning_signs": "ABCDE: Asymmetry, irregular Border, multiple Colors, Diameter >6mm, Evolution/changing",
        "risk_factors": "Fair skin, history of sunburns, many moles, family history",
        "action": "URGENT: See dermatologist within 1-2 weeks"
    },
    "Basal Cell Carcinoma": {
        "description": "Most common skin cancer. Grows slowly and rarely spreads, but can cause damage to surrounding tissue.",
        "warning_signs": "Pearl-like bump, pink growth, open sore that doesn't heal, shiny scar-like area",
        "risk_factors": "Chronic sun exposure, fair skin, age over 50",
        "action": "Schedule dermatology appointment (within 1 month)"
    },
    "Squamous Cell Carcinoma": {
        "description": "Second most common skin cancer. Can spread if untreated, but usually grows slowly.",
        "warning_signs": "Firm red nodule, flat lesion with scaly crust, sore that heals and re-opens",
        "risk_factors": "Long-term sun exposure, indoor tanning, weakened immune system",
        "action": "Schedule dermatology appointment (within 2-3 weeks)"
    },
    "Actinic Keratosis": {
        "description": "Precancerous skin growth caused by sun damage. May develop into squamous cell carcinoma if untreated.",
        "warning_signs": "Rough, scaly patch on sun-exposed skin, pink or red, feels like sandpaper",
        "risk_factors": "Fair skin, chronic sun exposure, age over 40",
        "action": "Monitor and consult dermatologist for possible treatment"
    },
    "Moles": {
        "description": "Common benign growths of melanocytes. Most are harmless, but some can become melanoma.",
        "warning_signs": "Normal moles are round, even-colored, and stable. Watch for changes!",
        "risk_factors": "Fair skin, many moles (50+), family history of melanoma",
        "action": "Regular self-checks, annual skin exam"
    },
    "Eczema": {
        "description": "Inflammatory skin condition causing itchy, red, dry skin. Not cancerous.",
        "warning_signs": "Itchy patches, dry scaly skin, redness, sometimes blisters",
        "risk_factors": "Family history of allergies/asthma, dry skin, stress",
        "action": "Use moisturizers, avoid triggers, see dermatologist if severe"
    },
    "Psoriasis": {
        "description": "Chronic autoimmune condition causing rapid skin cell buildup. Not contagious or cancerous.",
        "warning_signs": "Thick red patches with silvery scales, often on elbows/knees/scalp",
        "risk_factors": "Family history, stress, infections, certain medications",
        "action": "See dermatologist for treatment options"
    },
    "Acne": {
        "description": "Common skin condition when hair follicles clog with oil and dead skin cells.",
        "warning_signs": "Whiteheads, blackheads, pimples, cysts, usually on face/chest/back",
        "risk_factors": "Hormones, certain medications, diet, stress",
        "action": "Over-the-counter treatments, see dermatologist for severe cases"
    },
    "Rosacea": {
        "description": "Chronic skin condition causing facial redness and visible blood vessels.",
        "warning_signs": "Facial flushing, persistent redness, visible blood vessels, sometimes acne-like bumps",
        "risk_factors": "Fair skin, age 30-50, family history",
        "action": "Avoid triggers, see dermatologist for treatment"
    },
    "Warts": {
        "description": "Benign growths caused by HPV virus. Contagious but not cancerous.",
        "warning_signs": "Small rough bumps, may have black dots (clotted blood vessels)",
        "risk_factors": "Direct contact with virus, broken skin, weak immune system",
        "action": "Often resolve on their own, over-the-counter treatments available"
    },
    "Tinea": {
        "description": "Fungal skin infection (ringworm). Not caused by worms and not cancerous.",
        "warning_signs": "Ring-shaped rash with raised border, itchy, scaling",
        "risk_factors": "Contact with infected people/pets, damp environments",
        "action": "Antifungal creams, keep area clean and dry"
    },
    "Seborrheic Keratoses": {
        "description": "Common benign skin growths that appear with age. Sometimes mistaken for warts or skin cancer.",
        "warning_signs": "Waxy, stuck-on appearance, brown to black, often on chest/back/face",
        "risk_factors": "Age over 40, family history",
        "action": "Benign - no treatment needed unless irritated"
    },
    "Lichen": {
        "description": "Inflammatory skin condition. Not contagious or cancerous.",
        "warning_signs": "Purple, flat-topped bumps, often on wrists/ankles, itchy",
        "risk_factors": "Stress, certain medications, hepatitis C",
        "action": "See dermatologist for treatment options"
    },
    "Lupus": {
        "description": "Autoimmune disease where immune system attacks healthy tissue. Not contagious.",
        "warning_signs": "Butterfly-shaped rash across cheeks/nose, photosensitivity, joint pain",
        "risk_factors": "Female, age 15-44, African/Asian/Latino descent",
        "action": "See rheumatologist for proper diagnosis and treatment"
    },
    "Bullous": {
        "description": "Blistering skin conditions. Not cancerous but requires medical attention.",
        "warning_signs": "Large fluid-filled blisters on skin or mucous membranes",
        "risk_factors": "Autoimmune disorders, certain medications, infections",
        "action": "See dermatologist immediately for diagnosis"
    },
    "Candidiasis": {
        "description": "Fungal infection caused by Candida yeast. Not cancerous.",
        "warning_signs": "Red rash with satellite lesions, often in skin folds, itchy",
        "risk_factors": "Antibiotics, diabetes, weakened immune system, moisture",
        "action": "Antifungal creams, keep area dry, see doctor if persistent"
    },
    "Drug Eruption": {
        "description": "Skin reaction to medication. Can range from mild to severe.",
        "warning_signs": "Rash appearing days to weeks after starting new medication",
        "risk_factors": "New medications, multiple medications, previous drug allergies",
        "action": "Contact prescribing physician, do not stop medication without advice"
    },
    "Infestations/Bites": {
        "description": "Reactions to insect bites or parasitic infestations (scabies, lice).",
        "warning_signs": "Itchy bumps, burrows in skin (scabies), visible insects",
        "risk_factors": "Exposure to infested environments, close contact",
        "action": "Treat underlying cause, anti-itch medications"
    },
    "Vascular Tumors": {
        "description": "Growths of blood vessels. Usually benign (hemangiomas, angiomas).",
        "warning_signs": "Red or purple bumps that blanch with pressure",
        "risk_factors": "Often present at birth or develop with age",
        "action": "Usually harmless, consult dermatologist if changing/bleeding"
    },
    "Vasculitis": {
        "description": "Inflammation of blood vessels. Can affect skin and other organs.",
        "warning_signs": "Palpable purpura (raised purple spots), ulcers, livedo reticularis",
        "risk_factors": "Autoimmune diseases, infections, certain medications",
        "action": "URGENT: See doctor immediately - can affect internal organs"
    },
    "Vitiligo": {
        "description": "Autoimmune condition causing loss of skin pigment. Not contagious or cancerous.",
        "warning_signs": "White patches on skin, often symmetric, may affect hair color",
        "risk_factors": "Family history, other autoimmune diseases",
        "action": "See dermatologist for treatment options (cosmetic or medical)"
    },
    "Sun/Sunlight Damage": {
        "description": "Photoaging and sun-induced skin changes. Increases skin cancer risk.",
        "warning_signs": "Wrinkles, age spots, leathery texture, actinic keratoses",
        "risk_factors": "Chronic sun exposure, tanning beds, fair skin",
        "action": "Sun protection, regular skin exams, see dermatologist for concerning spots"
    },
    "Benign Tumors": {
        "description": "Non-cancerous growths that do not spread. Many types exist.",
        "warning_signs": "Slow-growing, well-defined borders, usually stable over time",
        "risk_factors": "Varies by type, often genetic or age-related",
        "action": "Monitor for changes, can be removed if bothersome"
    },
    "Normal": {
        "description": "No concerning findings detected in this image.",
        "warning_signs": "Skin appears healthy with no suspicious features",
        "risk_factors": "Continue sun protection and regular self-checks",
        "action": "Continue regular skin self-exams monthly, annual professional exam recommended"
    }
}

# ---------------- EDUCATIONAL TIPS ---------------- #
EDUCATIONAL_TIPS = [
    "Did you know? Melanoma can develop in existing moles or appear as new spots.",
    "The ABCDE rule helps remember what to look for: Asymmetry, Border, Color, Diameter, Evolution.",
    "Sunscreen with SPF 30+ reduces skin cancer risk by 40-50%.",
    "Check your skin monthly—look for anything new, changing, or unusual.",
    "Most skin cancers are caused by UV radiation from the sun. Wear protective clothing and seek shade.",
    "Melanoma caught early has a 99% 5-year survival rate.",
    "Even one severe sunburn in childhood doubles the risk of melanoma later in life.",
    "Basal cell carcinoma is the most common form of skin cancer but rarely spreads.",
    "Dermatologists recommend full-body skin exams annually, especially if you have risk factors.",
    "Not all moles are dangerous; benign moles have uniform color and regular borders.",
    "NOMA AI is open-source! Visit GitHub to build your own device.",
    "Skin cancer is the most common cancer in the US, but also one of the most preventable.",
    "The back is the most common location for melanoma in men; legs in women.",
    "People with dark skin can still get skin cancer, often on palms, soles, or under nails.",
    "Tanning beds increase melanoma risk by 75% when used before age 35.",
    "Your smartphone can help track moles over time - take photos monthly!",
    "Vitamin D from sun is important, but 10-15 minutes is enough for most people.",
    "Nail melanoma appears as dark streaks under fingernails or toenails.",
    "Early detection saves lives - know your skin and what's normal for you.",
    "Share NOMA AI with others - democratizing skin health awareness!"
]

# ---------------- Simple GPIO Controller ---------------- #
class SimpleLED:
    def __init__(self):
        self.gpio_available = False
        self.led_pins = {'red': 17, 'yellow': 27, 'green': 22}
        self.led_states = {'red': False, 'yellow': False, 'green': False}
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.gpio_available = True
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for pin in self.led_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            print("GPIO initialized successfully")
        except (ImportError, RuntimeError, PermissionError) as e:
            print(f"GPIO initialization failed, using mock: {e}")
            self.gpio_available = False

    def set_led(self, color, state):
        if color not in self.led_pins:
            return
        self.led_states[color] = state
        if self.gpio_available:
            pin = self.led_pins[color]
            if state:
                self.GPIO.output(pin, self.GPIO.HIGH)
            else:
                self.GPIO.output(pin, self.GPIO.LOW)
            print(f"GPIO {pin} ({color}): {'ON' if state else 'OFF'}")
        else:
            print(f"Mock LED {color}: {'ON' if state else 'OFF'}")

    def all_off(self):
        for color in self.led_pins:
            self.set_led(color, False)

    def cleanup(self):
        if self.gpio_available:
            try:
                print("Cleaning up GPIO...")
                self.all_off()
                self.GPIO.cleanup()
                print("GPIO cleanup complete")
            except Exception as e:
                print(f"Error during GPIO cleanup: {e}")

led_controller = SimpleLED()

def set_leds(red=False, yellow=False, green=False):
    led_controller.set_led('red', red)
    led_controller.set_led('yellow', yellow)
    led_controller.set_led('green', green)

def turn_off_leds():
    led_controller.all_off()

# ---------------- Health Passport ---------------- #
class HealthPassport:
    def __init__(self):
        self.history_dir = "/home/havil/noma_ai"
        self.history_file = os.path.join(self.history_dir, "health_passport.json")
        self._ensure_dir()
        self._ensure_file()

    def _ensure_dir(self):
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

    def _ensure_file(self):
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump([], f)

    def save_assessment(self, results, image_path=None):
        record = {
            'timestamp': datetime.now().isoformat(),
            'ai_prediction': results.get('cnn_prediction', 'unknown'),
            'confidence': results.get('cnn_confidence', 0.0),
            'abcde_score': results.get('abcde_score', 0),
            'patient_score': results.get('patient_score', 0),
            'total_risk': results.get('total_risk', 0),
            'led_color': results.get('led_color', 'GREEN'),
            'image_path': image_path
        }
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            history.append(record)
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print("Assessment saved to health passport")
        except Exception as e:
            print(f"Failed to save health passport: {e}")

health_passport = HealthPassport()

# ---------------- ENHANCED Clinical Feature Extractor ---------------- #
class ClinicalFeatureExtractor:
    @staticmethod
    def calculate_asymmetry(image):
        try:
            img = np.array(image.convert('L'))
            h, w = img.shape
            left = img[:, :w//2]
            right = img[:, w//2:]
            right_flipped = np.fliplr(right)
            min_h = min(left.shape[0], right_flipped.shape[0])
            min_w = min(left.shape[1], right_flipped.shape[1])
            left = left[:min_h, :min_w]
            right_flipped = right_flipped[:min_h, :min_w]
            diff = np.mean(np.abs(left - right_flipped))
            return min(diff / 50.0, 1.0)
        except Exception as e:
            print(f"Asymmetry calculation error: {e}")
            return 0.5

    @staticmethod
    def calculate_asymmetry_score(image):
        """Return a more detailed asymmetry score with explanation"""
        try:
            img = np.array(image.convert('L'))
            h, w = img.shape
            left = img[:, :w//2]
            right = img[:, w//2:]
            right_flipped = np.fliplr(right)
            min_h = min(left.shape[0], right_flipped.shape[0])
            min_w = min(left.shape[1], right_flipped.shape[1])
            left = left[:min_h, :min_w]
            right_flipped = right_flipped[:min_h, :min_w]
            diff = np.mean(np.abs(left - right_flipped))
            score = min(diff / 50.0, 1.0)
            
            if score > 0.7:
                explanation = "Highly asymmetrical - the two halves differ significantly"
            elif score > 0.4:
                explanation = "Moderately asymmetrical - some difference between halves"
            else:
                explanation = "Largely symmetrical - halves are similar"
            
            return score, explanation
        except Exception as e:
            return 0.5, "Could not calculate asymmetry"

    @staticmethod
    def calculate_border_irregularity(image):
        try:
            img = np.array(image.convert('L'))
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.5
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if area == 0:
                return 0.5
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            irregularity = max(0, min(1, 1 - circularity))
            return irregularity
        except Exception as e:
            print(f"Border irregularity calculation error: {e}")
            return 0.5

    @staticmethod
    def calculate_border_score(image):
        """Return border irregularity score with explanation"""
        try:
            img = np.array(image.convert('L'))
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5, "Could not detect lesion border"
            
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if area == 0:
                return 0.5, "Border area too small"
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            irregularity = max(0, min(1, 1 - circularity))
            
            if irregularity > 0.7:
                explanation = "Highly irregular border - ragged, notched, or poorly defined"
            elif irregularity > 0.4:
                explanation = "Moderately irregular border - some unevenness detected"
            else:
                explanation = "Smooth, well-defined border"
            
            return irregularity, explanation
        except Exception as e:
            return 0.5, "Could not calculate border irregularity"

    @staticmethod
    def calculate_color_uniformity(image):
        try:
            img = np.array(image)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            std_h = np.std(hsv[:, :, 0])
            std_s = np.std(hsv[:, :, 1])
            std_v = np.std(hsv[:, :, 2])
            total_std = (std_h + std_s + std_v) / 3
            return min(total_std / 50.0, 1.0)
        except Exception as e:
            print(f"Color uniformity calculation error: {e}")
            return 0.5

    @staticmethod
    def analyze_color_distribution(image):
        """Detailed color analysis"""
        try:
            img = np.array(image)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            mean_h = np.mean(hsv[:, :, 0])
            std_h = np.std(hsv[:, :, 0])
            mean_s = np.mean(hsv[:, :, 1])
            std_s = np.std(hsv[:, :, 1])
            
            h_bins = np.histogram(hsv[:, :, 0], bins=10)[0]
            distinct_estimate = np.sum(h_bins > (np.max(h_bins) * 0.1))
            
            color_score = min(std_h / 50.0, 1.0)
            
            if color_score > 0.6:
                explanation = f"Multiple colors detected - {distinct_estimate}+ distinct shades present"
            elif color_score > 0.3:
                explanation = f"Some color variation - approximately {distinct_estimate} distinct shades"
            else:
                explanation = "Uniform color throughout lesion"
            
            return color_score, explanation, distinct_estimate
        except Exception as e:
            return 0.5, "Could not analyze color", 1

    @staticmethod
    def estimate_diameter(image, reference_mm=10):
        try:
            img = np.array(image.convert('L'))
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            pixels_per_mm = 6.4
            diameter_mm = max(w, h) / pixels_per_mm
            return diameter_mm
        except Exception as e:
            print(f"Diameter estimation error: {e}")
            return 0

    @staticmethod
    def generate_clinical_report(features):
        """Generate a human-readable clinical report"""
        report = []
        
        asymmetry_score, asymmetry_exp = features.get('asymmetry', (0.5, ""))
        border_score, border_exp = features.get('border', (0.5, ""))
        color_score, color_exp, color_count = features.get('color', (0.5, "", 1))
        diameter_mm = features.get('diameter_mm', 0)
        
        report.append("=== CLINICAL FEATURE ANALYSIS ===\n")
        
        if asymmetry_score > 0.6:
            report.append(f"⚠️ ASYMMETRY: {asymmetry_exp}")
        else:
            report.append(f"✓ SYMMETRY: {asymmetry_exp}")
        
        if border_score > 0.6:
            report.append(f"⚠️ BORDER: {border_exp}")
        else:
            report.append(f"✓ BORDER: {border_exp}")
        
        if color_score > 0.5:
            report.append(f"⚠️ COLOR: {color_exp}")
        else:
            report.append(f"✓ COLOR: {color_exp}")
        
        if diameter_mm > 6:
            report.append(f"⚠️ DIAMETER: {diameter_mm:.1f}mm (>6mm threshold)")
        elif diameter_mm > 0:
            report.append(f"✓ DIAMETER: {diameter_mm:.1f}mm (within normal range)")
        else:
            report.append("? DIAMETER: Could not measure accurately")
        
        risk_factors = sum([
            asymmetry_score > 0.5,
            border_score > 0.5,
            color_score > 0.4,
            diameter_mm > 6
        ])
        
        report.append("\n=== SUMMARY ===\n")
        if risk_factors >= 3:
            report.append("HIGH RISK: Multiple concerning features detected")
        elif risk_factors >= 2:
            report.append("MODERATE RISK: Some concerning features present")
        elif risk_factors >= 1:
            report.append("LOW RISK: Minor features detected, monitor for changes")
        else:
            report.append("VERY LOW RISK: No concerning features detected")
        
        return "\n".join(report)

# ---------------- STEP-BY-STEP CLINICAL ASSESSOR (ENHANCED WITH COMPACT TOP NAV) ---------------- #
class StepByStepClinicalAssessor(QtWidgets.QDialog):
    def __init__(self, parent=None, cnn_prediction="", cnn_confidence=0.0):
        super().__init__(parent)
        self.cnn_prediction = cnn_prediction
        self.cnn_confidence = cnn_confidence
        self.parent_app = parent

        self.abcde_answers = {
            'asymmetry': False,
            'border': False,
            'color': 'single',
            'diameter': 'small',
            'evolution': 'none'
        }
        self.patient_data = {
            'age': 40,
            'skin_type': 2,
            'family_history': False,
            'sunburn_history': False,
            'itchy': False,
            'painful': False,
            'bleeding': False,
            'sudden_onset': False,
            'slow_onset': False,
            'recurrence': False,
            'sun_exposure': False,
            'family_skin_cancer': False,
            'personal_history': False,
            'immune_suppressed': False
        }

        self.current_step = 0
        self.total_steps = 8
        self.current_widgets = []

        self.initUI()
        if self.parent_app:
            self.parent_app.start_yellow_blinking_for_dialog()
        self.show_step(0)

    def initUI(self):
        self.setWindowTitle("Clinical Assessment Wizard")
        self.setFixedSize(800, 600)
        self.setStyleSheet("""
            QDialog { background-color: #b8fcbf; }
            QLabel { font-size: 16px; margin: 5px; }
            QPushButton { font-size: 14px; font-weight: bold; padding: 8px 16px; margin: 3px; border-radius: 8px; min-width: 80px; }
            QProgressBar { height: 15px; border: 2px solid #94ffed; border-radius: 10px; background-color: white; }
            QProgressBar::chunk { background-color: #94ffed; border-radius: 8px; }
            QRadioButton { font-size: 18px; margin: 8px; padding: 8px; spacing: 8px; background-color: transparent; }
            QRadioButton::indicator { width: 20px; height: 20px; }
            QRadioButton:hover { color: #00695c; font-weight: bold; }
            QSpinBox { font-size: 18px; padding: 8px; min-height: 30px; border: 2px solid #94ffed; border-radius: 8px; background-color: white; }
            QComboBox { font-size: 16px; padding: 10px; min-height: 35px; border: 2px solid #94ffed; border-radius: 8px; background-color: white; }
            QComboBox:hover { border: 2px solid #00695c; }
            QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 30px; border-left: 2px solid #94ffed; border-radius: 0px; }
            QComboBox QAbstractItemView { font-size: 16px; background-color: white; selection-background-color: #94ffed; selection-color: #00695c; border: 2px solid #94ffed; outline: none; }
            QCheckBox { font-size: 18px; margin: 8px; padding: 8px; spacing: 8px; }
            QCheckBox::indicator { width: 20px; height: 20px; }
            QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        self.title_label = QLabel("CLINICAL ASSESSMENT")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; padding: 5px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        # Navigation bar (BACK and NEXT side by side at top)
        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setSpacing(15)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        self.back_button = QPushButton("← BACK")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #ffd794;
                color: #654700;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ffe7a8;
            }
        """)
        self.back_button.clicked.connect(self.previous_step)
        self.back_button.setVisible(False)
        nav_layout.addWidget(self.back_button)
        
        nav_layout.addStretch(1)
        
        self.next_button = QPushButton("NEXT →")
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #94ffed;
                color: #00695c;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #a8fff0;
            }
            QPushButton:pressed {
                background-color: #80dfd0;
            }
        """)
        self.next_button.clicked.connect(self.next_step)
        nav_layout.addWidget(self.next_button)
        
        main_layout.addWidget(nav_bar)

        # Step label and progress bar container
        step_container = QWidget()
        step_layout = QVBoxLayout(step_container)
        step_layout.setSpacing(5)
        step_layout.setContentsMargins(0, 0, 0, 0)
        
        self.step_label = QLabel("Step 1 of 8")
        self.step_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00695c;")
        self.step_label.setAlignment(Qt.AlignCenter)
        step_layout.addWidget(self.step_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.total_steps)
        self.progress_bar.setValue(1)
        step_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(step_container)

        # Question container (scrollable area for content)
        self.question_scroll = QScrollArea()
        self.question_scroll.setWidgetResizable(True)
        self.question_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.question_container = QWidget()
        self.question_layout = QVBoxLayout(self.question_container)
        self.question_layout.setSpacing(12)
        self.question_layout.setContentsMargins(10, 10, 10, 10)
        self.question_layout.addStretch()
        
        self.question_scroll.setWidget(self.question_container)
        main_layout.addWidget(self.question_scroll)

        # Cancel button at bottom
        self.cancel_button = QPushButton("CANCEL ASSESSMENT")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9494;
                color: #690000;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ffa8a8;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_assessment)
        main_layout.addWidget(self.cancel_button)
        
        self.setLayout(main_layout)

    def clear_question_area(self):
        # Clear the question layout but keep the stretch
        while self.question_layout.count():
            item = self.question_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Re-add stretch
        self.question_layout.addStretch()
        self.current_widgets.clear()
        QApplication.processEvents()

    def show_step(self, step):
        self.save_answers()
        self.clear_question_area()
        self.current_step = step
        self.progress_bar.setValue(step + 1)
        self.step_label.setText(f"Step {step + 1} of {self.total_steps}")
        self.back_button.setVisible(step > 0)
        
        if step == self.total_steps - 1:
            self.next_button.setText("CALCULATE →")
        else:
            self.next_button.setText("NEXT →")

        if step == 0:
            self.show_asymmetry_step()
        elif step == 1:
            self.show_border_step()
        elif step == 2:
            self.show_color_step()
        elif step == 3:
            self.show_diameter_step()
        elif step == 4:
            self.show_evolution_step()
        elif step == 5:
            self.show_enhanced_context_step()
        elif step == 6:
            self.show_patient_info_step()
        elif step == 7:
            self.show_summary_step()
        QApplication.processEvents()

    def show_asymmetry_step(self):
        self.title_label.setText("A - ASYMMETRY")
        
        question = QLabel("Is the lesion asymmetrical?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)

        description = QLabel("Asymmetry means if you draw a line through the middle, the two halves don't match.")
        description.setStyleSheet("font-size: 14px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, description)
        self.current_widgets.append(description)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: rgba(255,255,255,0.3); }")
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(8)
        self.asymmetry_yes = QRadioButton("YES - Asymmetrical")
        self.asymmetry_no = QRadioButton("NO - Symmetrical")
        self.asymmetry_no.setChecked(True)
        for rb in [self.asymmetry_yes, self.asymmetry_no]:
            rb.setStyleSheet("QRadioButton { font-size: 18px; padding: 10px; background-color: white; border: 1px solid #94ffed; border-radius: 8px; margin: 3px; } QRadioButton:hover { background-color: #e8fff8; border: 2px solid #00695c; } QRadioButton:checked { background-color: #94ffed; border: 2px solid #00695c; font-weight: bold; }")
        option_layout.addWidget(self.asymmetry_yes)
        option_layout.addWidget(self.asymmetry_no)
        self.question_layout.insertWidget(self.question_layout.count() - 1, option_group)
        self.current_widgets.extend([option_group, self.asymmetry_yes, self.asymmetry_no])

    def show_border_step(self):
        self.title_label.setText("B - BORDER")
        
        question = QLabel("Is the border irregular, ragged, or notched?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)

        description = QLabel("Irregular borders look like a coastline with bays, not a smooth, round circle.")
        description.setStyleSheet("font-size: 14px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, description)
        self.current_widgets.append(description)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: rgba(255,255,255,0.3); }")
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(8)
        self.border_yes = QRadioButton("YES - Irregular border")
        self.border_no = QRadioButton("NO - Smooth border")
        self.border_no.setChecked(True)
        for rb in [self.border_yes, self.border_no]:
            rb.setStyleSheet("QRadioButton { font-size: 18px; padding: 10px; background-color: white; border: 1px solid #94ffed; border-radius: 8px; margin: 3px; } QRadioButton:hover { background-color: #e8fff8; border: 2px solid #00695c; } QRadioButton:checked { background-color: #94ffed; border: 2px solid #00695c; font-weight: bold; }")
        option_layout.addWidget(self.border_yes)
        option_layout.addWidget(self.border_no)
        self.question_layout.insertWidget(self.question_layout.count() - 1, option_group)
        self.current_widgets.extend([option_group, self.border_yes, self.border_no])

    def show_color_step(self):
        self.title_label.setText("C - COLOR")
        
        question = QLabel("How many colors does the lesion have?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)

        description = QLabel("More colors = higher risk. Look for shades of brown, black, red, white, or blue.")
        description.setStyleSheet("font-size: 14px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, description)
        self.current_widgets.append(description)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: rgba(255,255,255,0.3); }")
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(8)
        self.color_single = QRadioButton("Single color (brown, tan, or black)")
        self.color_two = QRadioButton("2-3 colors")
        self.color_many = QRadioButton("Many different colors")
        self.color_single.setChecked(True)
        for rb in [self.color_single, self.color_two, self.color_many]:
            rb.setStyleSheet("QRadioButton { font-size: 18px; padding: 10px; background-color: white; border: 1px solid #94ffed; border-radius: 8px; margin: 3px; } QRadioButton:hover { background-color: #e8fff8; border: 2px solid #00695c; } QRadioButton:checked { background-color: #94ffed; border: 2px solid #00695c; font-weight: bold; }")
        option_layout.addWidget(self.color_single)
        option_layout.addWidget(self.color_two)
        option_layout.addWidget(self.color_many)
        self.question_layout.insertWidget(self.question_layout.count() - 1, option_group)
        self.current_widgets.extend([option_group, self.color_single, self.color_two, self.color_many])

    def show_diameter_step(self):
        self.title_label.setText("D - DIAMETER")
        
        question = QLabel("What is the size of the lesion?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)

        description = QLabel("Measure the widest part. A pencil eraser is about 6mm.")
        description.setStyleSheet("font-size: 14px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, description)
        self.current_widgets.append(description)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: rgba(255,255,255,0.3); }")
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(8)
        self.diameter_small = QRadioButton("Small (less than 6mm)")
        self.diameter_medium = QRadioButton("Medium (6-10mm)")
        self.diameter_large = QRadioButton("Large (more than 10mm)")
        self.diameter_small.setChecked(True)
        for rb in [self.diameter_small, self.diameter_medium, self.diameter_large]:
            rb.setStyleSheet("QRadioButton { font-size: 18px; padding: 10px; background-color: white; border: 1px solid #94ffed; border-radius: 8px; margin: 3px; } QRadioButton:hover { background-color: #e8fff8; border: 2px solid #00695c; } QRadioButton:checked { background-color: #94ffed; border: 2px solid #00695c; font-weight: bold; }")
        option_layout.addWidget(self.diameter_small)
        option_layout.addWidget(self.diameter_medium)
        option_layout.addWidget(self.diameter_large)
        self.question_layout.insertWidget(self.question_layout.count() - 1, option_group)
        self.current_widgets.extend([option_group, self.diameter_small, self.diameter_medium, self.diameter_large])

    def show_evolution_step(self):
        self.title_label.setText("E - EVOLUTION")
        
        question = QLabel("Has the lesion changed over time?")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)

        description = QLabel("Recent changes in size, shape, or color are a major warning sign!")
        description.setStyleSheet("font-size: 14px; font-style: italic; color: #d32f2f; font-weight: bold; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, description)
        self.current_widgets.append(description)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 10px; padding-top: 12px; background-color: rgba(255,255,255,0.3); }")
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(8)
        self.evolution_no = QRadioButton("NO CHANGE - Stable for years")
        self.evolution_slow = QRadioButton("SLOW CHANGE - Over months/years")
        self.evolution_fast = QRadioButton("RAPID CHANGE - Weeks/months")
        self.evolution_no.setChecked(True)
        for rb in [self.evolution_no, self.evolution_slow, self.evolution_fast]:
            rb.setStyleSheet("QRadioButton { font-size: 18px; padding: 10px; background-color: white; border: 1px solid #94ffed; border-radius: 8px; margin: 3px; } QRadioButton:hover { background-color: #e8fff8; border: 2px solid #00695c; } QRadioButton:checked { background-color: #94ffed; border: 2px solid #00695c; font-weight: bold; }")
        option_layout.addWidget(self.evolution_no)
        option_layout.addWidget(self.evolution_slow)
        option_layout.addWidget(self.evolution_fast)
        self.question_layout.insertWidget(self.question_layout.count() - 1, option_group)
        self.current_widgets.extend([option_group, self.evolution_no, self.evolution_slow, self.evolution_fast])

    def show_enhanced_context_step(self):
        """Enhanced clinical context questions beyond standard ABCDE"""
        self.title_label.setText("CLINICAL CONTEXT")
        
        question = QLabel("Additional Information")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)
        
        description = QLabel("These details help differentiate between conditions")
        description.setStyleSheet("font-size: 14px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, description)
        self.current_widgets.append(description)
        
        context_widget = QtWidgets.QWidget()
        context_layout = QVBoxLayout(context_widget)
        context_layout.setSpacing(10)
        context_layout.setContentsMargins(5, 5, 5, 5)
        
        # Symptom-based questions
        symptom_group = QtWidgets.QGroupBox("Symptoms")
        symptom_group.setStyleSheet("QGroupBox { font-size: 15px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 8px; padding-top: 12px; background-color: white; }")
        symptom_layout = QVBoxLayout(symptom_group)
        
        self.itchy_check = QCheckBox("Is the lesion itchy?")
        self.painful_check = QCheckBox("Is it painful or tender?")
        self.bleeding_check = QCheckBox("Does it bleed easily?")
        
        for cb in [self.itchy_check, self.painful_check, self.bleeding_check]:
            cb.setStyleSheet("QCheckBox { font-size: 15px; padding: 8px; background-color: white; border: 1px solid #94ffed; border-radius: 6px; margin: 2px; } QCheckBox:hover { background-color: #e8fff8; } QCheckBox:checked { background-color: #94ffed; } QCheckBox::indicator { width: 18px; height: 18px; }")
            symptom_layout.addWidget(cb)
        
        context_layout.addWidget(symptom_group)
        
        # Onset pattern
        onset_group = QtWidgets.QGroupBox("Onset & Pattern")
        onset_group.setStyleSheet("QGroupBox { font-size: 15px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 8px; padding-top: 12px; background-color: white; }")
        onset_layout = QVBoxLayout(onset_group)
        
        self.sudden_check = QCheckBox("Appeared suddenly (within days/weeks)")
        self.slow_check = QCheckBox("Developed slowly (months/years)")
        self.recur_check = QCheckBox("Has occurred before (recurrent)")
        
        for cb in [self.sudden_check, self.slow_check, self.recur_check]:
            cb.setStyleSheet("QCheckBox { font-size: 15px; padding: 8px; background-color: white; border: 1px solid #94ffed; border-radius: 6px; margin: 2px; } QCheckBox:hover { background-color: #e8fff8; } QCheckBox:checked { background-color: #94ffed; } QCheckBox::indicator { width: 18px; height: 18px; }")
            onset_layout.addWidget(cb)
        
        context_layout.addWidget(onset_group)
        
        # Risk factors
        risk_context_group = QtWidgets.QGroupBox("Risk Factors")
        risk_context_group.setStyleSheet("QGroupBox { font-size: 15px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 8px; padding-top: 12px; background-color: white; }")
        risk_context_layout = QVBoxLayout(risk_context_group)
        
        self.sun_exposure_check = QCheckBox("History of significant sun exposure / tanning bed use")
        self.family_skin_cancer_check = QCheckBox("Family history of skin cancer")
        self.personal_history_check = QCheckBox("Personal history of skin cancer")
        self.immune_check = QCheckBox("Weakened immune system (medication/condition)")
        
        for cb in [self.sun_exposure_check, self.family_skin_cancer_check, self.personal_history_check, self.immune_check]:
            cb.setStyleSheet("QCheckBox { font-size: 15px; padding: 8px; background-color: white; border: 1px solid #94ffed; border-radius: 6px; margin: 2px; } QCheckBox:hover { background-color: #e8fff8; } QCheckBox:checked { background-color: #94ffed; } QCheckBox::indicator { width: 18px; height: 18px; }")
            risk_context_layout.addWidget(cb)
        
        context_layout.addWidget(risk_context_group)
        
        self.question_layout.insertWidget(self.question_layout.count() - 1, context_widget)
        self.current_widgets.extend([context_widget, symptom_group, onset_group, risk_context_group,
                                     self.itchy_check, self.painful_check, self.bleeding_check,
                                     self.sudden_check, self.slow_check, self.recur_check,
                                     self.sun_exposure_check, self.family_skin_cancer_check,
                                     self.personal_history_check, self.immune_check])

    def show_patient_info_step(self):
        self.title_label.setText("PATIENT INFORMATION")
        
        question = QLabel("Tell us about the patient")
        question.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, question)
        self.current_widgets.append(question)
        
        info_widget = QtWidgets.QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setSpacing(10)
        info_layout.setContentsMargins(5, 5, 5, 5)

        age_group = QtWidgets.QGroupBox("Age")
        age_group.setStyleSheet("QGroupBox { font-size: 15px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 8px; padding-top: 12px; background-color: white; }")
        age_layout = QHBoxLayout(age_group)
        age_label = QLabel("Patient age:")
        age_label.setStyleSheet("font-size: 16px; padding: 5px;")
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setRange(1, 120)
        self.age_spinbox.setValue(40)
        self.age_spinbox.setStyleSheet("QSpinBox { font-size: 18px; padding: 8px; min-height: 35px; border: 2px solid #94ffed; border-radius: 6px; background-color: white; }")
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_spinbox)
        age_layout.addStretch(1)
        info_layout.addWidget(age_group)

        skin_group = QtWidgets.QGroupBox("Skin Type (Fitzpatrick Scale)")
        skin_group.setStyleSheet("QGroupBox { font-size: 15px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 8px; padding-top: 12px; background-color: white; }")
        skin_layout = QVBoxLayout(skin_group)
        skin_description = QLabel("Select the skin type that best matches:")
        skin_description.setStyleSheet("font-size: 13px; color: #666; margin: 5px;")
        skin_description.setWordWrap(True)
        skin_layout.addWidget(skin_description)
        self.skin_combo = QComboBox()
        self.skin_combo.addItems([
            "Type I - Always burns, never tans",
            "Type II - Usually burns, tans minimally",
            "Type III - Sometimes burns, tans gradually",
            "Type IV - Rarely burns, tans well",
            "Type V - Very rarely burns, tans easily",
            "Type VI - Never burns"
        ])
        self.skin_combo.setCurrentIndex(2)
        self.skin_combo.setStyleSheet("QComboBox { font-size: 14px; padding: 10px; min-height: 40px; border: 2px solid #94ffed; border-radius: 6px; background-color: white; }")
        skin_layout.addWidget(self.skin_combo)
        info_layout.addWidget(skin_group)

        self.question_layout.insertWidget(self.question_layout.count() - 1, info_widget)
        self.current_widgets.extend([info_widget, age_group, skin_group, self.age_spinbox, self.skin_combo])

    def show_summary_step(self):
        self.title_label.setText("SUMMARY")
        self.save_answers()
        
        summary_text = QLabel("Review your answers:")
        summary_text.setStyleSheet("font-size: 22px; font-weight: bold; color: #00695c; margin: 5px;")
        summary_text.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, summary_text)
        self.current_widgets.append(summary_text)

        summary_display = QTextEdit()
        summary_display.setReadOnly(True)
        summary_display.setMaximumHeight(350)
        summary_display.setStyleSheet("QTextEdit { background-color: white; border: 2px solid #94ffed; border-radius: 10px; padding: 12px; font-size: 14px; }")

        summary_html = "<h3 style='color: #00695c;'>ABCDE Assessment:</h3>"
        summary_html += f"<p><b>A. Asymmetry:</b> {'YES' if self.abcde_answers['asymmetry'] else 'NO'}</p>"
        summary_html += f"<p><b>B. Border:</b> {'YES' if self.abcde_answers['border'] else 'NO'}</p>"
        summary_html += f"<p><b>C. Color:</b> {self.abcde_answers['color'].title()}</p>"
        summary_html += f"<p><b>D. Diameter:</b> {self.abcde_answers['diameter'].title()}</p>"
        summary_html += f"<p><b>E. Evolution:</b> {self.abcde_answers['evolution'].title().replace('_', ' ')}</p>"

        summary_html += "<h3 style='color: #00695c; margin-top: 15px;'>Clinical Context:</h3>"
        summary_html += f"<p><b>Itchy:</b> {'YES' if self.patient_data.get('itchy', False) else 'NO'}</p>"
        summary_html += f"<p><b>Painful:</b> {'YES' if self.patient_data.get('painful', False) else 'NO'}</p>"
        summary_html += f"<p><b>Bleeding:</b> {'YES' if self.patient_data.get('bleeding', False) else 'NO'}</p>"
        summary_html += f"<p><b>Sudden Onset:</b> {'YES' if self.patient_data.get('sudden_onset', False) else 'NO'}</p>"
        summary_html += f"<p><b>Slow Onset:</b> {'YES' if self.patient_data.get('slow_onset', False) else 'NO'}</p>"
        summary_html += f"<p><b>Recurrence:</b> {'YES' if self.patient_data.get('recurrence', False) else 'NO'}</p>"
        summary_html += f"<p><b>Sun Exposure:</b> {'YES' if self.patient_data.get('sun_exposure', False) else 'NO'}</p>"
        summary_html += f"<p><b>Family Skin Cancer:</b> {'YES' if self.patient_data.get('family_skin_cancer', False) else 'NO'}</p>"
        summary_html += f"<p><b>Personal History:</b> {'YES' if self.patient_data.get('personal_history', False) else 'NO'}</p>"
        summary_html += f"<p><b>Immune Suppressed:</b> {'YES' if self.patient_data.get('immune_suppressed', False) else 'NO'}</p>"

        summary_html += "<h3 style='color: #00695c; margin-top: 15px;'>Patient Information:</h3>"
        summary_html += f"<p><b>Age:</b> {self.patient_data['age']}</p>"
        skin_types = [
            "Type I - Always burns, never tans",
            "Type II - Usually burns, tans minimally",
            "Type III - Sometimes burns, tans gradually",
            "Type IV - Rarely burns, tans well",
            "Type V - Very rarely burns, tans easily",
            "Type VI - Never burns"
        ]
        summary_html += f"<p><b>Skin Type:</b> {skin_types[self.patient_data['skin_type']]}</p>"

        summary_display.setHtml(summary_html)
        self.question_layout.insertWidget(self.question_layout.count() - 1, summary_display)
        self.current_widgets.append(summary_display)

        note = QLabel("Click 'CALCULATE' to generate your comprehensive risk assessment.")
        note.setStyleSheet("font-size: 13px; font-style: italic; color: #666; margin-top: 8px;")
        note.setAlignment(Qt.AlignCenter)
        self.question_layout.insertWidget(self.question_layout.count() - 1, note)
        self.current_widgets.append(note)

    def save_answers(self):
        if self.current_step == 0 and hasattr(self, 'asymmetry_yes'):
            self.abcde_answers['asymmetry'] = self.asymmetry_yes.isChecked()
        elif self.current_step == 1 and hasattr(self, 'border_yes'):
            self.abcde_answers['border'] = self.border_yes.isChecked()
        elif self.current_step == 2 and hasattr(self, 'color_single'):
            if self.color_single.isChecked():
                self.abcde_answers['color'] = 'single'
            elif self.color_two.isChecked():
                self.abcde_answers['color'] = 'two'
            else:
                self.abcde_answers['color'] = 'many'
        elif self.current_step == 3 and hasattr(self, 'diameter_small'):
            if self.diameter_small.isChecked():
                self.abcde_answers['diameter'] = 'small'
            elif self.diameter_medium.isChecked():
                self.abcde_answers['diameter'] = 'medium'
            else:
                self.abcde_answers['diameter'] = 'large'
        elif self.current_step == 4 and hasattr(self, 'evolution_no'):
            if self.evolution_no.isChecked():
                self.abcde_answers['evolution'] = 'none'
            elif self.evolution_slow.isChecked():
                self.abcde_answers['evolution'] = 'slow'
            else:
                self.abcde_answers['evolution'] = 'fast'
        elif self.current_step == 5 and hasattr(self, 'itchy_check'):
            self.patient_data['itchy'] = self.itchy_check.isChecked()
            self.patient_data['painful'] = self.painful_check.isChecked()
            self.patient_data['bleeding'] = self.bleeding_check.isChecked()
            self.patient_data['sudden_onset'] = self.sudden_check.isChecked()
            self.patient_data['slow_onset'] = self.slow_check.isChecked()
            self.patient_data['recurrence'] = self.recur_check.isChecked()
            self.patient_data['sun_exposure'] = self.sun_exposure_check.isChecked()
            self.patient_data['family_skin_cancer'] = self.family_skin_cancer_check.isChecked()
            self.patient_data['personal_history'] = self.personal_history_check.isChecked()
            self.patient_data['immune_suppressed'] = self.immune_check.isChecked()
        elif self.current_step == 6 and hasattr(self, 'age_spinbox'):
            self.patient_data['age'] = self.age_spinbox.value()
            self.patient_data['skin_type'] = self.skin_combo.currentIndex()

    def previous_step(self):
        if self.current_step > 0:
            self.show_step(self.current_step - 1)

    def next_step(self):
        self.save_answers()
        if self.current_step < self.total_steps - 1:
            self.show_step(self.current_step + 1)
        else:
            self.calculate_results()

    def calculate_results(self):
        if self.parent_app:
            self.parent_app.stop_yellow_blinking()

        abcde_score = 0
        if self.abcde_answers['asymmetry']:
            abcde_score += 1
        if self.abcde_answers['border']:
            abcde_score += 1
        if self.abcde_answers['color'] == 'two':
            abcde_score += 1
        elif self.abcde_answers['color'] == 'many':
            abcde_score += 2
        if self.abcde_answers['diameter'] in ['medium', 'large']:
            abcde_score += 1
        if self.abcde_answers['evolution'] == 'slow':
            abcde_score += 1
        elif self.abcde_answers['evolution'] == 'fast':
            abcde_score += 2

        patient_score = 0
        if self.patient_data['age'] > 50:
            patient_score += 1
        if self.patient_data['skin_type'] < 2:
            patient_score += 1
        if self.patient_data.get('family_skin_cancer', False):
            patient_score += 1
        if self.patient_data.get('sun_exposure', False):
            patient_score += 1
        if self.patient_data.get('personal_history', False):
            patient_score += 1

        if "melanoma" in self.cnn_prediction.lower() or "carcinoma" in self.cnn_prediction.lower():
            cnn_weight = 3
        elif self.cnn_prediction.lower() == "normal":
            cnn_weight = 0
        else:
            cnn_weight = 1

        total_risk = (
            (abcde_score / 8.0) * 40 +
            (patient_score / 5.0) * 20 +
            (cnn_weight * self.cnn_confidence) * 40
        )

        self.final_results = {
            'abcde_score': abcde_score,
            'patient_score': patient_score,
            'total_risk': total_risk,
            'cnn_prediction': self.cnn_prediction,
            'cnn_confidence': self.cnn_confidence
        }

        if total_risk >= 70:
            self.final_results['led_color'] = "RED"
            self.final_results['recommendation'] = """URGENT: Dermatology referral (within 1-2 weeks)\nDo not delay evaluation\nMonitor for any changes\nAvoid sun exposure"""
        elif total_risk >= 40:
            self.final_results['led_color'] = "YELLOW"
            self.final_results['recommendation'] = """Schedule dermatology appointment (4-6 weeks)\nMonitor monthly for changes\nPractice sun protection\nUse SPF 30+ daily"""
        else:
            self.final_results['led_color'] = "GREEN"
            self.final_results['recommendation'] = """Continue regular self-checks\nAnnual skin examination recommended\nPractice sun safety\nUse SPF 15+ daily"""

        if self.parent_app:
            self.parent_app.show_green_completion_pattern()
        self.accept()

    def cancel_assessment(self):
        reply = QMessageBox.question(self, 'Cancel', 'Cancel assessment? All answers will be lost.',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.parent_app:
                self.parent_app.stop_yellow_blinking()
            self.reject()

    def reject(self):
        if self.parent_app:
            self.parent_app.stop_yellow_blinking()
        super().reject()

# ---------------- Camera Thread ---------------- #
class CameraThread(QThread):
    frame_ready = pyqtSignal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.running = True
        self.latest_frame = None
        self.picam2 = None

    def run(self):
        try:
            print("Starting camera...")
            self.picam2 = Picamera2()
            try:
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    controls={"FrameRate": 30}
                )
                self.picam2.configure(config)
                print("Camera configured with RGB888 format")
            except:
                print("Falling back to simple configuration")
                config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
                self.picam2.configure(config)
            self.picam2.start()
            print("Camera started successfully")
            while self.running:
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        elif frame.shape[2] == 3:
                            h, w, _ = frame.shape
                            sample = frame[h//2, w//2]
                            if sample[2] > sample[0] + 30:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.latest_frame = frame.copy()
                        h, w, ch = frame.shape
                        qt_image = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
                        self.frame_ready.emit(qt_image)
                    time.sleep(0.03)
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"Camera initialization error: {e}")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            while self.running:
                h, w, ch = dummy_frame.shape
                qt_image = QtGui.QImage(dummy_frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
                time.sleep(0.1)

    def get_latest_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        if self.picam2:
            try:
                self.picam2.stop()
            except:
                pass
        self.wait(1000)

# ---------------- Main App ---------------- #
class NomaAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        turn_off_leds()

        self.blink_timer = None
        self.blink_state = False
        self.blink_count = 0
        self.max_blinks = 20

        self.classes = [
            "Acne", "Actinic Keratosis", "Benign Tumors", "Bullous",
            "Candidiasis", "Drug Eruption", "Eczema", "Infestations/Bites",
            "Lichen", "Lupus", "Moles", "Psoriasis", "Rosacea",
            "Seborrheic Keratoses", "Melanoma",
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
        self.is_classifying = False

        self.initUI()
        self.load_model()
        self.start_camera()
        self.education_timer = None

    def initUI(self):
        self.setWindowTitle("NOMA AI - Operation Oracle")
        self.showFullScreen()
        screen_geometry = QApplication.primaryScreen().geometry()
        self.screen_height = screen_geometry.height()
        print(f"Screen height: {self.screen_height}")

        self.setStyleSheet("""
            QMainWindow { background-color: #b8fcbf; }
            QWidget { background-color: #b8fcbf; }
            QScrollArea { border: none; background-color: #b8fcbf; }
        """)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        central_widget = QtWidgets.QWidget()
        self.scroll_area.setWidget(central_widget)
        self.setCentralWidget(self.scroll_area)

        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("NOMA AI | Operation Oracle")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #00695c;
                padding: 15px;
                background-color: #94ffed;
                border: 3px solid #80dfd0;
                border-radius: 15px;
                margin: 5px;
            }
        """)
        layout.addWidget(title_label)

        # Subtitle
        subtitle = QLabel("Democratizing Early Detection | Explainable AI | Open-Source")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #00695c; margin-bottom: 5px;")
        layout.addWidget(subtitle)

        # Camera preview
        self.image_label = QLabel("Loading camera feed...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setMaximumSize(400, 300)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 3px solid #94ffed;
                border-radius: 10px;
                padding: 5px;
                margin: 5px;
            }
        """)
        camera_container = QtWidgets.QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.image_label)
        layout.addWidget(camera_container)

        # Classify button
        self.classify_button = QPushButton("CAPTURE AND ANALYZE")
        self.classify_button.setMinimumHeight(100)
        self.classify_button.setObjectName("classify_button")
        self.classify_button.setStyleSheet("""
            QPushButton#classify_button {
                font-size: 24px;
                font-weight: bold;
                padding: 20px 15px;
                background-color: #94ffed;
                color: #00695c;
                border: 4px solid #80dfd0;
                border-radius: 20px;
                margin: 10px;
            }
            QPushButton#classify_button:hover { background-color: #a8fff0; border: 4px solid #94ffed; }
            QPushButton#classify_button:pressed { background-color: #80dfd0; color: #004d40; }
            QPushButton#classify_button:disabled { background-color: #c8fcf5; color: #80a09c; }
        """)
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)
        layout.addWidget(self.classify_button)

        # LED Guide button
        self.led_guide_button = QPushButton("LED STATUS GUIDE")
        self.led_guide_button.setMinimumHeight(80)
        self.led_guide_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                padding: 15px 10px;
                background-color: #ffd794;
                color: #654700;
                border: 4px solid #dfc080;
                border-radius: 20px;
                margin: 10px;
            }
            QPushButton:hover { background-color: #ffe7a8; border: 4px solid #ffd794; }
            QPushButton:pressed { background-color: #dfc080; color: #402d00; }
        """)
        self.led_guide_button.clicked.connect(self.show_led_guide)
        layout.addWidget(self.led_guide_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #94ffed;
                border-radius: 10px;
                text-align: center;
                background-color: white;
                margin: 5px;
            }
            QProgressBar::chunk {
                background-color: #94ffed;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Results area
        self.results_label = QLabel("")
        self.results_label.setAlignment(Qt.AlignLeft)
        self.results_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                background-color: #defcee;
                padding: 15px;
                border: 2px solid #94ffed;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        self.results_label.setWordWrap(True)
        self.results_label.setMinimumHeight(150)
        layout.addWidget(self.results_label)

        # Educational tip banner
        self.tip_label = QLabel("Tip: " + random.choice(EDUCATIONAL_TIPS))
        self.tip_label.setAlignment(Qt.AlignCenter)
        self.tip_label.setWordWrap(True)
        self.tip_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-style: italic;
                color: #00695c;
                background-color: #e0f7fa;
                padding: 8px;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        layout.addWidget(self.tip_label)

        # Analysis image with Grad-CAM explanation
        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)
        
        self.analysis_label = QLabel("")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.analysis_label.setMinimumSize(400, 300)
        self.analysis_label.setStyleSheet("""
            QLabel {
                background-color: #defcee;
                border: 2px solid #94ffed;
                border-radius: 10px;
                padding: 5px;
                margin: 5px;
            }
        """)
        analysis_layout.addWidget(self.analysis_label)
        
        # Grad-CAM explanation text
        self.gradcam_explanation = QLabel("")
        self.gradcam_explanation.setWordWrap(True)
        self.gradcam_explanation.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #555;
                padding: 5px;
                margin: 0px 5px 5px 5px;
                background-color: #f0f8f0;
                border-radius: 8px;
            }
        """)
        analysis_layout.addWidget(self.gradcam_explanation)
        
        layout.addWidget(analysis_container)

        # Open Source / Documentation button
        self.docs_button = QPushButton("BUILD YOUR OWN | OPEN-SOURCE DOCS")
        self.docs_button.setMinimumHeight(60)
        self.docs_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 12px 10px;
                background-color: #c8e6df;
                color: #00695c;
                border: 3px solid #94ffed;
                border-radius: 15px;
                margin: 5px;
            }
            QPushButton:hover { background-color: #b2dfdb; }
        """)
        self.docs_button.clicked.connect(self.show_documentation)
        layout.addWidget(self.docs_button)

        # Shutdown button
        self.shutdown_button = QPushButton("SHUTDOWN DEVICE")
        self.shutdown_button.setMinimumHeight(80)
        self.shutdown_button.setObjectName("shutdown_button")
        self.shutdown_button.setStyleSheet("""
            QPushButton#shutdown_button {
                font-size: 20px;
                font-weight: bold;
                padding: 15px 10px;
                background-color: #ff9494;
                color: #690000;
                border: 4px solid #df8080;
                border-radius: 20px;
                margin: 10px;
            }
            QPushButton#shutdown_button:hover { background-color: #ffa8a8; border: 4px solid #ff9494; }
            QPushButton#shutdown_button:pressed { background-color: #df8080; color: #400000; }
        """)
        self.shutdown_button.clicked.connect(self.shutdown_device)
        layout.addWidget(self.shutdown_button)

        # Disclaimer
        disclaimer_label = QLabel("*Not a medical diagnosis. For educational use only.*\nOpen-source hardware/software | Democratizing skin health | Operation Oracle")
        disclaimer_label.setAlignment(Qt.AlignCenter)
        disclaimer_label.setWordWrap(True)
        disclaimer_label.setStyleSheet("font-size: 11px; font-style: italic; color: #666; padding: 5px; margin: 5px;")
        layout.addWidget(disclaimer_label)

        layout.addStretch(1)

        # Start tip rotation timer
        self.tip_timer = QTimer()
        self.tip_timer.timeout.connect(self.rotate_tip)
        self.tip_timer.start(30000)

    def rotate_tip(self):
        self.tip_label.setText("Tip: " + random.choice(EDUCATIONAL_TIPS))

    def show_documentation(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("NOMA AI | Open-Source Documentation")
        dialog.setMinimumSize(600, 500)
        dialog.setStyleSheet("QDialog { background-color: #b8fcbf; }")
        
        layout = QVBoxLayout(dialog)
        
        title = QLabel("Operation Oracle | Open-Source Documentation")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00695c; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("background-color: white; border: 2px solid #94ffed; border-radius: 10px; padding: 15px; font-size: 14px;")
        
        docs_html = """
        <h2 style='color: #00695c;'>Democratizing Skin Health</h2>
        <p>NOMA AI is completely open-source - hardware, software, and documentation. You can build your own device!</p>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Enhanced Explainable Features:</h3>
        <ul>
            <li><b>Grad-CAM Heatmaps:</b> Visualize which areas influenced the AI's decision</li>
            <li><b>Clinical Feature Extraction:</b> Automated asymmetry, border, color, and diameter analysis</li>
            <li><b>ABCDE Integration:</b> Gold-standard clinical assessment combined with AI</li>
            <li><b>Comprehensive Risk Factors:</b> Patient history, symptoms, and onset patterns</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Hardware Requirements:</h3>
        <ul>
            <li>Raspberry Pi 4 (4GB+)</li>
            <li>Arducam 16MP IMX519 (or any compatible camera)</li>
            <li>Waveshare 5" Touchscreen (800x480)</li>
            <li>RGB LEDs (3x - Red, Yellow, Green) with 220 Ohm resistors</li>
            <li>Power bank (5V/3A+)</li>
            <li>3D printed enclosure (files available online)</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Software Stack:</h3>
        <ul>
            <li>Raspberry Pi OS Lite (Bookworm)</li>
            <li>Python 3.9+ with PyQt5 for GUI</li>
            <li>TensorFlow Lite for model inference</li>
            <li>Picamera2 for camera control</li>
            <li>OpenCV for image processing</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Model Training:</h3>
        <ul>
            <li>Dataset: 12,900 images across 24 skin conditions</li>
            <li>Architecture: MobileNetV3 (ImageNet pre-trained)</li>
            <li>Input size: 224x224 pixels</li>
            <li>Framework: TensorFlow 2.18</li>
            <li>Deployment: Optimized .tflite for edge inference</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Educational Resources:</h3>
        <ul>
            <li><b>Online Course:</b> "Building Medical AI for Edge Devices" (Coming soon)</li>
            <li><b>GitHub Repository:</b> Complete source code and schematics</li>
            <li><b>YouTube Tutorials:</b> Step-by-step assembly guide</li>
            <li><b>Community Forum:</b> Share your builds and improvements</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>How Grad-CAM Works:</h3>
        <p>The heatmap visualization shows which areas of the image most influenced the AI's decision. Red/yellow areas indicate regions that strongly suggested the predicted condition, while blue/green areas had less influence. This helps you understand what the AI is "looking at" when making its prediction.</p>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Contribute:</h3>
        <ul>
            <li>Improve the model with your own data</li>
            <li>Translate the UI to other languages</li>
            <li>Design better enclosures</li>
            <li>Create educational content</li>
        </ul>
        
        <p style='margin-top: 20px;'><b>Visit:</b> <a href='https://github.com/havil/noma-ai'>github.com/havil/noma-ai</a></p>
        <p><b>Contact:</b> noma.operation.oracle@gmail.com</p>
        """
        
        text_edit.setHtml(docs_html)
        layout.addWidget(text_edit)
        
        close_button = QPushButton("CLOSE")
        close_button.setMinimumHeight(50)
        close_button.setStyleSheet("font-size: 16px; font-weight: bold; padding: 12px; background-color: #94ffed; color: #00695c; border-radius: 15px; margin: 10px;")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec_()

    def show_led_guide(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("LED Status Guide")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("QDialog { background-color: #b8fcbf; } QTextEdit { background-color: white; border: 2px solid #94ffed; border-radius: 10px; padding: 15px; font-size: 14px; }")
        layout = QVBoxLayout(dialog)
        title = QLabel("LED STATUS GUIDE")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00695c; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h3>YELLOW BLINKING:</h3><ul><li>System is analyzing your image</li><li>Please wait while processing completes</li><li>This takes about 5-10 seconds</li></ul>
        <h3>SOLID RED:</h3><ul><li>MALIGNANT detection</li><li>High risk potential</li><li>Consult dermatologist promptly</li><li>Examples: Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma</li></ul>
        <h3>SOLID YELLOW:</h3><ul><li>BENIGN detection</li><li>Moderate risk or uncertain</li><li>Monitor regularly</li><li>Examples: Moles, Eczema, Psoriasis, Acne</li></ul>
        <h3>SOLID GREEN:</h3><ul><li>NORMAL skin</li><li>Low risk</li><li>Continue regular self-checks</li></ul>
        <h3>ADDITIONAL NOTES:</h3><ul><li>All LEDs turn off automatically after 8 seconds</li><li>Low confidence results will show yellow LED with recommendation to retake image</li><li>Ensure good lighting and clear focus for best results</li><li>Position lesion clearly in the center of the camera view</li></ul>
        """)
        layout.addWidget(text_edit)
        exit_button = QPushButton("CLOSE GUIDE")
        exit_button.setMinimumHeight(50)
        exit_button.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 12px; background-color: #94ffed; color: #00695c; border: 3px solid #80dfd0; border-radius: 15px; margin: 10px; }")
        exit_button.clicked.connect(dialog.accept)
        layout.addWidget(exit_button)
        dialog.exec_()

    def start_yellow_blinking_for_dialog(self):
        self.stop_yellow_blinking()
        self.blink_state = False
        self.blink_count = 0
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink_yellow)
        self.blink_timer.start(500)

    def _blink_yellow(self):
        self.blink_count += 1
        if self.blink_count > self.max_blinks:
            self.stop_yellow_blinking()
            return
        self.blink_state = not self.blink_state
        set_leds(yellow=self.blink_state)

    def stop_yellow_blinking(self):
        if self.blink_timer:
            self.blink_timer.stop()
            self.blink_timer = None
        set_leds(yellow=False)
        self.blink_count = 0

    def show_green_completion_pattern(self):
        self.stop_yellow_blinking()
        set_leds(green=True)
        QTimer.singleShot(300, lambda: set_leds(green=False))
        QTimer.singleShot(600, lambda: set_leds(green=True))
        QTimer.singleShot(900, lambda: set_leds(green=False))
        QTimer.singleShot(1200, lambda: set_leds(green=True))
        QTimer.singleShot(1500, lambda: set_leds(green=False))

    def generate_grad_cam_heatmap(self, image, predicted_class, confidence):
        try:
            img_array = np.array(image.resize((224, 224)))
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img_array, 0.5, heatmap, 0.5, 0)
            return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return image

    def get_disease_info_html(self, disease_name):
        if disease_name in DISEASE_INFO:
            info = DISEASE_INFO[disease_name]
            return f"""
            <hr>
            <h3 style='color: #00695c;'>About {disease_name}:</h3>
            <p><b>Description:</b> {info['description']}</p>
            <p><b>Warning Signs:</b> {info['warning_signs']}</p>
            <p><b>Risk Factors:</b> {info['risk_factors']}</p>
            <p><b>Recommended Action:</b> {info['action']}</p>
            <hr>
            """
        return ""

    def classify_image(self):
        if self.is_classifying:
            return
        self.is_classifying = True
        self.classify_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.results_label.setText("Starting analysis...")
        QApplication.processEvents()

        try:
            frame = self.camera_thread.get_latest_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "No camera feed")
                return
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(frame)
            img_array = self.preprocess_image(image)

            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            normal_index = self.classes.index("Normal") if "Normal" in self.classes else -1
            if normal_index >= 0 and predictions[0][normal_index] > 0.3:
                sorted_indices = np.argsort(predictions[0])[::-1]
                if sorted_indices[0] == normal_index and len(sorted_indices) > 1:
                    second_confidence = predictions[0][sorted_indices[1]]
                    if second_confidence > 0.3:
                        predictions[0][normal_index] *= 0.85
                        predictions[0] = predictions[0] / np.sum(predictions[0])
                        class_index = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])

            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3 = [(self.classes[i], predictions[0][i]) for i in top3_indices]
            top3_text = "\n".join([f"{i+1}. {cls} ({conf:.1%})" for i, (cls, conf) in enumerate(top3)])

            sorted_probs = np.sort(predictions[0])[::-1]
            uncertainty = 1 - (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.5

            predicted_class = self.classes[class_index]

            # Extract enhanced clinical features
            asymmetry_score, asymmetry_exp = ClinicalFeatureExtractor.calculate_asymmetry_score(image)
            border_score, border_exp = ClinicalFeatureExtractor.calculate_border_score(image)
            color_score, color_exp, color_count = ClinicalFeatureExtractor.analyze_color_distribution(image)
            diameter_mm = ClinicalFeatureExtractor.estimate_diameter(image)

            features = {
                'asymmetry': asymmetry_score,
                'border': border_score,
                'color_uniformity': color_score,
                'diameter_mm': diameter_mm
            }

            # Generate enhanced clinical report
            clinical_report = ClinicalFeatureExtractor.generate_clinical_report({
                'asymmetry': (asymmetry_score, asymmetry_exp),
                'border': (border_score, border_exp),
                'color': (color_score, color_exp, color_count),
                'diameter_mm': diameter_mm
            })

            feature_importance = (
                f"Asymmetry: {asymmetry_score:.2f} - {asymmetry_exp}\n"
                f"Border irregularity: {border_score:.2f} - {border_exp}\n"
                f"Color variation: {color_score:.2f} - {color_exp}\n"
                f"Diameter: {diameter_mm:.1f} mm"
            )

            # Handle low confidence
            if uncertainty > 0.8:
                QMessageBox.warning(self, "High Uncertainty",
                    f"Analysis uncertainty is high ({uncertainty:.0%}). Please retake with better lighting and focus.")
                self.set_leds_timed(False, True, False)
                self.results_label.setText(f"High uncertainty ({uncertainty:.0%}). Please retake image with better lighting.")
                return

            if confidence < 0.5:
                QMessageBox.warning(self, "Low Confidence",
                    f"Confidence too low ({confidence:.1%}).\nPlease retake with better lighting or positioning.")
                self.set_leds_timed(False, True, False)
                self.results_label.setText(f"Low confidence ({confidence:.1%}). Please retake image with better lighting.")
                return

            dialog = StepByStepClinicalAssessor(self, predicted_class, confidence)

            if dialog.exec_():
                results = dialog.final_results

                # Generate Grad-CAM
                heatmap = self.generate_grad_cam_heatmap(image, predicted_class, confidence)
                heatmap_array = np.array(heatmap)
                heatmap_qimage = QtGui.QImage(heatmap_array.data, heatmap_array.shape[1], heatmap_array.shape[0],
                                              heatmap_array.strides[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(heatmap_qimage).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.analysis_label.setPixmap(pixmap)
                
                self.gradcam_explanation.setText(
                    "🔬 GRAD-CAM EXPLANATION: Red/yellow areas show where the AI focused most. "
                    "These regions had the strongest influence on the prediction of " + predicted_class + ". "
                    "Compare this heatmap with the clinical features below to understand the AI's reasoning."
                )

                led_color = results['led_color']
                if led_color == "RED":
                    self.set_leds_timed(True, False, False)
                elif led_color == "YELLOW":
                    self.set_leds_timed(False, True, False)
                else:
                    self.set_leds_timed(False, False, True)

                disease_info = self.get_disease_info_html(results['cnn_prediction'])

                result_text = f"""
COMPREHENSIVE ANALYSIS COMPLETE

AI DETECTION:
- Condition: {results['cnn_prediction']}
- Confidence: {results['cnn_confidence']:.1%}

TOP ALTERNATIVES:
{top3_text}

ENHANCED CLINICAL FEATURE ANALYSIS:
{feature_importance}

{clinical_report}

CLINICAL ASSESSMENT:
- ABCDE Score: {results['abcde_score']}/8
- Patient Risk: {results['patient_score']}/5
- Total Risk: {results['total_risk']:.0f}/100

RISK LEVEL: {led_color}

RECOMMENDATION:
{results['recommendation']}

{disease_info}

🔬 UNDERSTANDING YOUR RESULTS:
- The heatmap above shows which areas most influenced the AI
- The clinical features match what dermatologists look for
- Compare both to understand the complete assessment

*This is a screening tool only. Always consult a healthcare professional.*
"""
                self.results_label.setText(result_text)

                health_passport.save_assessment(results)

            else:
                self.results_label.setText("Assessment cancelled.")
                turn_off_leds()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.results_label.setText(f"Error: {str(e)}")
        finally:
            self.is_classifying = False
            self.classify_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_camera_feed)
        self.camera_thread.start()

    def update_camera_feed(self, qt_image):
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.repaint()

    def load_model(self):
        try:
            model_path = '/home/havil/noma_ai/nomaai_model.tflite'
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded successfully")
            print(f"Input dtype: {self.input_details[0]['dtype']}")
            print(f"Input shape: {self.input_details[0]['shape']}")
            self.classify_button.setEnabled(True)
        except Exception as e:
            print(f"Model error: {e}")
            self.results_label.setText(f"Model error: {str(e)}")

    def preprocess_image(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image.resize((224, 224)), dtype=np.float32)

        expected_dtype = self.input_details[0]['dtype']
        expected_shape = self.input_details[0]['shape']

        if expected_dtype == np.float32:
            img_array = (img_array / 127.5) - 1.0
        elif expected_dtype == np.uint8:
            img_array = img_array.astype(np.uint8)
        else:
            img_array = (img_array / 127.5) - 1.0

        if len(expected_shape) == 4 and len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        elif len(expected_shape) == 3 and len(img_array.shape) == 4:
            img_array = img_array.squeeze(0)

        img_array = img_array.astype(expected_dtype)
        return img_array

    def set_leds(self, red=False, yellow=False, green=False):
        set_leds(red=red, yellow=yellow, green=green)

    def set_leds_timed(self, red=False, yellow=False, green=False):
        set_leds(red=red, yellow=yellow, green=green)
        QTimer.singleShot(8000, turn_off_leds)

    def turn_off_leds(self):
        turn_off_leds()

    def shutdown_device(self):
        reply = QMessageBox.question(self, "Shutdown", "Shut down the device?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            led_controller.cleanup()
            os.system("sudo shutdown now")

    def closeEvent(self, event):
        print("Closing application...")
        self.stop_yellow_blinking()
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if hasattr(self, 'tip_timer'):
            self.tip_timer.stop()
        led_controller.cleanup()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setOverrideCursor(Qt.BlankCursor)
    ex = NomaAIApp()
    ex.show()
    app.aboutToQuit.connect(led_controller.cleanup)
    sys.exit(app.exec_())
