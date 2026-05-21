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
import sqlite3
import hashlib
import shutil
from datetime import datetime, timedelta
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QPushButton, QApplication,
                             QMessageBox, QProgressBar, QMainWindow, QScrollArea,
                             QWidget, QHBoxLayout, QTextEdit, QRadioButton,
                             QSpinBox, QComboBox, QCheckBox, QGroupBox, QTabWidget,
                             QListWidget, QListWidgetItem, QDialog, QLineEdit, QSlider, QDialogButtonBox,
                             QInputDialog, QGridLayout, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPixmap
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
    "Infestations/Bites": {
        "description": "Reactions to insect bites or parasitic infestations (scabies, lice). Not cancerous but may cause discomfort.",
        "warning_signs": "Itchy bumps, burrows in skin (scabies), visible insects, localized redness",
        "risk_factors": "Exposure to infested environments, close contact with infected individuals, outdoor activities",
        "action": "Treat underlying cause with appropriate medication, anti-itch creams available over-the-counter"
    },
    "Normal": {
        "description": "No concerning findings detected in this image.",
        "warning_signs": "Skin appears healthy with no suspicious features",
        "risk_factors": "Continue sun protection and regular self-checks",
        "action": "Continue regular skin self-exams monthly, annual professional exam recommended"
    }
}

# Add more disease info dynamically
for disease in ["Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Eczema", "Psoriasis", "Acne", "Rosacea", "Warts", "Tinea", "Drug Eruption"]:
    if disease not in DISEASE_INFO:
        DISEASE_INFO[disease] = {
            "description": f"{disease} is a skin condition that should be evaluated by a healthcare professional.",
            "warning_signs": "Visible skin changes, discomfort, or unusual appearance",
            "risk_factors": "Varies by condition",
            "action": "Consult healthcare provider for proper diagnosis and treatment"
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

# ---------------- LONGITUDINAL TRACKING DATABASE ---------------- #
# Get the actual home directory path (works for both pi and havil)
HOME_DIR = os.path.expanduser("~")
# Use the shared folder for cross-device syncing
SYNC_FOLDER = "/opt/oracle_share"
os.makedirs(SYNC_FOLDER, exist_ok=True)

DB_PATH = os.path.join(HOME_DIR, "noma_longitudinal.db")
TRACKED_IMAGES_DIR = os.path.join(HOME_DIR, "noma_ai", "tracked_lesions")
os.makedirs(TRACKED_IMAGES_DIR, exist_ok=True)

def init_tracking_db():
    """Initialize the SQLite database for longitudinal tracking"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lesions (
        lesion_id TEXT PRIMARY KEY,
        first_seen TEXT,
        body_location TEXT,
        patient_name TEXT,
        feature_descriptors BLOB,
        feature_count INTEGER
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        lesion_id TEXT,
        timestamp TEXT,
        image_path TEXT,
        prediction TEXT,
        confidence REAL,
        abcde_scores TEXT,
        risk_level TEXT,
        match_count INTEGER,
        ita_score REAL,
        skin_tone TEXT,
        FOREIGN KEY (lesion_id) REFERENCES lesions(lesion_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Longitudinal tracking database initialized at {DB_PATH}")
    print(f"Sync folder: {SYNC_FOLDER}")

init_tracking_db()

# ---------------- ORB FEATURE EXTRACTOR ---------------- #
def extract_lesion_features(image_array, n_features=500):
    """Creates a unique fingerprint of a lesion using ORB with pyramid scale invariance"""
    try:
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        gray = cv2.resize(gray, (224, 224))
        
        orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        if descriptors is not None and len(descriptors) > 0:
            return descriptors.tobytes(), len(keypoints)
        else:
            return None, 0
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None, 0

def compare_lesions(descriptors1_bytes, descriptors2_bytes, match_threshold=35):
    """Compares two lesion fingerprints using ORB with Lowe's ratio test"""
    try:
        if descriptors1_bytes is None or descriptors2_bytes is None:
            return 0.0, 0, False
        
        desc1 = np.frombuffer(descriptors1_bytes, dtype=np.uint8)
        desc2 = np.frombuffer(descriptors2_bytes, dtype=np.uint8)
        
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0, 0, False
        
        if len(desc1) % 32 != 0 or len(desc2) % 32 != 0:
            return 0.0, 0, False
            
        desc1 = desc1.reshape((-1, 32))
        desc2 = desc2.reshape((-1, 32))
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        raw_match_count = len(good_matches)
        
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0, 0, False
            
        match_score = raw_match_count / min(len(desc1), len(desc2))
        is_same = raw_match_count >= match_threshold
        
        dynamic_threshold = int(0.15 * min(len(desc1), len(desc2)))
        effective_threshold = max(match_threshold, dynamic_threshold)
        is_same_dynamic = raw_match_count >= effective_threshold
        
        is_same = is_same and is_same_dynamic
        
        return match_score, raw_match_count, is_same
        
    except Exception as e:
        print(f"Comparison error: {e}")
        return 0.0, 0, False

def detect_changes(old_scan, new_scan):
    """Compare two scans of the same lesion and report changes"""
    changes = []
    
    if new_scan.get('confidence', 0) - old_scan.get('confidence', 0) > 0.2:
        changes.append(f"AI confidence increased by {(new_scan['confidence'] - old_scan['confidence'])*100:.0f}%")
    
    if new_scan.get('prediction', '') != old_scan.get('prediction', ''):
        changes.append(f"Diagnosis changed from {old_scan.get('prediction', 'unknown')} to {new_scan.get('prediction', 'unknown')}")
    
    risk_order = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2, 'URGENT': 3}
    old_risk = risk_order.get(old_scan.get('risk_level', 'LOW'), 0)
    new_risk = risk_order.get(new_scan.get('risk_level', 'LOW'), 0)
    
    if new_risk > old_risk:
        changes.append(f"Risk level increased from {old_scan.get('risk_level', 'LOW')} to {new_scan.get('risk_level', 'LOW')}")
    
    return changes

def sync_scan_to_shared_folder(scan_data):
    """Save scan result to synced folder so other device can see it"""
    try:
        filename = os.path.join(SYNC_FOLDER, f"noma_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
        
        scan_data['source_device'] = 'NOMA_AI'
        scan_data['scan_type'] = 'skin'
        scan_data['timestamp'] = datetime.now().isoformat()
        
        with open(filename, 'w') as f:
            json.dump(scan_data, f, indent=2)
        
        print(f"Scan synced to shared folder: {filename}")
        return True
    except Exception as e:
        print(f"Sync error: {e}")
        return False

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
        self.history_dir = os.path.join(HOME_DIR, "noma_ai")
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
            'image_path': image_path,
            'ita_score': results.get('ita_score', 0),
            'skin_tone': results.get('skin_tone', 'Unknown')
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

# ---------------- ITA-based Preprocessing and Grad-CAM ---------------- #
class ITAPreprocessor:
    """Handles ITA-based skin tone detection and adaptive preprocessing"""
    
    @staticmethod
    def calculate_ita(image_array):
        """
        Calculate Individual Typology Angle for skin tone estimation
        ITA = arctan((L* - 50) / b*) in degrees
        Higher ITA = lighter skin, Lower ITA = darker skin
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0].astype(np.float32)
            b = lab[:, :, 2].astype(np.float32)
            
            # ITA formula: arctan((L* - 50) / b*) in degrees
            mean_L = np.mean(L) * (100.0 / 255.0)
            mean_b = np.mean(b) - 128.0
            
            # Avoid division by zero
            if abs(mean_b) < 0.01:
                mean_b = 0.01 if mean_b >= 0 else -0.01
            
            ita_rad = np.arctan((mean_L - 50.0) / mean_b)
            ita_deg = ita_rad * 180.0 / np.pi
            
            # Dermatology standard thresholds (Del Bino 2015)
            if ita_deg > 55:
                skin_tone = "Very Light"
                contrast_boost = 1.0
                bias_risk = "Low"
            elif ita_deg > 41:
                skin_tone = "Light"
                contrast_boost = 1.0
                bias_risk = "Low"
            elif ita_deg > 28:
                skin_tone = "Intermediate"
                contrast_boost = 1.2
                bias_risk = "Medium"
            elif ita_deg > 10:
                skin_tone = "Tan"
                contrast_boost = 1.3
                bias_risk = "Medium"
            elif ita_deg > -30:
                skin_tone = "Brown"
                contrast_boost = 1.5
                bias_risk = "High"
            else:
                skin_tone = "Dark"
                contrast_boost = 1.8
                bias_risk = "Highest"
            
            return ita_deg, skin_tone, contrast_boost, bias_risk
        except Exception as e:
            print(f"ITA calculation error: {e}")
            return 0.0, "Unknown", 1.0, "Unknown"
    
    @staticmethod
    def apply_adaptive_contrast(image_array, boost_factor):
        """Apply adaptive contrast enhancement based on skin tone"""
        if boost_factor <= 1.0:
            return image_array
        
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=boost_factor, tileGridSize=(8, 8))
        enhanced_L = clahe.apply(L)
        
        lab[:, :, 0] = enhanced_L
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced


class GradCAMVisualizer:
    """Creates a heatmap around the detected lesion area using contour detection."""
    
    @staticmethod
    def detect_lesion_contour(image_array):
        """Detect the lesion contour in the image to create focused heatmap."""
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                x, y, w, h = cv2.boundingRect(largest_contour)
                return mask, (x, y, w, h), cv2.contourArea(largest_contour)
            else:
                h, w = gray.shape
                center_mask = np.zeros(gray.shape, dtype=np.uint8)
                center_mask[h//4:3*h//4, w//4:3*w//4] = 255
                return center_mask, (w//4, h//4, w//2, h//2), (w*h)//4
                
        except Exception as e:
            print(f"Contour detection error: {e}")
            h, w = image_array.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
            return mask, (w//4, h//4, w//2, h//2), (w*h)//4
    
    @staticmethod
    def generate_heatmap(image_array, predicted_class, confidence):
        """Generate Grad-CAM style heatmap focused on the detected lesion area."""
        try:
            mask, bbox, area = GradCAMVisualizer.detect_lesion_contour(image_array)
            x, y, w, h = bbox
            
            h_img, w_img = image_array.shape[:2]
            heatmap = np.zeros((h_img, w_img), dtype=np.float32)
            center_x = x + w // 2
            center_y = y + h // 2
            
            for i in range(h_img):
                for j in range(w_img):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    max_dist = max(h, w) / 2 if max(h, w) > 0 else 1
                    heatmap[i, j] = np.exp(-(dist**2) / (2 * (max_dist/2)**2))
            
            heatmap = heatmap * (mask > 0).astype(np.float32)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            heatmap = np.power(heatmap, 1.5 - confidence * 0.5)
            return heatmap, bbox
            
        except Exception as e:
            print(f"Heatmap generation error: {e}")
            h, w = image_array.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            center_y, center_x = h // 2, w // 2
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    max_dist = max(h, w) / 2 if max(h, w) > 0 else 1
                    heatmap[i, j] = np.exp(-(dist**2) / (2 * (max_dist/2)**2))
            return heatmap, (w//4, h//4, w//2, h//2)
    
    @staticmethod
    def overlay_heatmap(original_image, heatmap, alpha=0.5):
        """Overlay heatmap on original image."""
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return blended

# ---------------- Clinical Feature Extractor ---------------- #
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
        report = []
        
        asymmetry_score, asymmetry_exp = features.get('asymmetry', (0.5, ""))
        border_score, border_exp = features.get('border', (0.5, ""))
        color_score, color_exp, color_count = features.get('color', (0.5, "", 1))
        diameter_mm = features.get('diameter_mm', 0)
        
        report.append("=== CLINICAL FEATURE ANALYSIS ===\n")
        
        if asymmetry_score > 0.6:
            report.append(f"ASYMMETRY: {asymmetry_exp}")
        else:
            report.append(f"SYMMETRY: {asymmetry_exp}")
        
        if border_score > 0.6:
            report.append(f"BORDER: {border_exp}")
        else:
            report.append(f"BORDER: {border_exp}")
        
        if color_score > 0.5:
            report.append(f"COLOR: {color_exp}")
        else:
            report.append(f"COLOR: {color_exp}")
        
        if diameter_mm > 6:
            report.append(f"DIAMETER: {diameter_mm:.1f}mm (>6mm threshold)")
        elif diameter_mm > 0:
            report.append(f"DIAMETER: {diameter_mm:.1f}mm (within normal range)")
        else:
            report.append("DIAMETER: Could not measure accurately")
        
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


# ---------------- CAMERA THREAD - WORKING PERFECTLY ---------------- #
class CameraThread(QThread):
    frame_ready = pyqtSignal(QtGui.QImage)

    def __init__(self, parent_app=None):
        super().__init__()
        self.running = True
        self.latest_frame = None
        self.picam2 = None
        self.parent_app = parent_app

    def run(self):
        try:
            print("Starting camera...")
            self.picam2 = Picamera2()
            
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            print("Camera started successfully")
            time.sleep(1.0)
            
            while self.running:
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        elif frame.shape[2] == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        self.latest_frame = frame.copy()
                        h, w, ch = frame.shape
                        bytes_per_line = ch * w
                        qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        self.frame_ready.emit(qt_image)
                    time.sleep(0.033)
                except Exception as e:
                    print(f"Frame capture error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"Camera initialization error: {e}")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy_frame[:, :] = [100, 150, 100]
            cv2.putText(dummy_frame, "Camera Error", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            while self.running:
                h, w, ch = dummy_frame.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(dummy_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
                time.sleep(0.1)

    def get_latest_frame(self):
        return self.latest_frame

    def stop(self):
        self.running = False
        if self.picam2:
            try:
                self.picam2.stop()
                print("Camera stopped")
            except Exception as e:
                print(f"Error stopping camera: {e}")
        self.wait(1000)


# ---------------- BODY LOCATION SELECTION DIALOG ---------------- #
class BodyLocationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_location = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Select Lesion Location")
        self.setModal(True)
        self.setFixedSize(500, 400)
        self.setStyleSheet("""
            QDialog { background-color: #b8fcbf; }
            QPushButton { 
                font-size: 16px; 
                font-weight: bold; 
                padding: 15px; 
                margin: 5px;
                border-radius: 10px;
                background-color: #94ffed;
                color: #00695c;
            }
            QPushButton:hover { background-color: #a8fff0; }
            QLabel { font-size: 18px; font-weight: bold; color: #00695c; margin: 10px; }
        """)
        
        layout = QVBoxLayout()
        
        title = QLabel("Where is this lesion located?")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(10)
        
        locations = [
            "Face", "Scalp", "Neck", "Chest", "Abdomen", "Back",
            "Shoulder", "Upper Arm", "Forearm", "Hand", "Upper Leg", 
            "Lower Leg", "Foot", "Other"
        ]
        
        row = 0
        col = 0
        for location in locations:
            btn = QPushButton(location)
            btn.clicked.connect(lambda checked, loc=location: self.select_location(loc))
            grid_layout.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        layout.addWidget(grid_widget)
        
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setStyleSheet("background-color: #ff9494; color: #690000;")
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(cancel_btn)
        
        self.setLayout(layout)
    
    def select_location(self, location):
        self.selected_location = location
        self.accept()
    
    def get_location(self):
        return self.selected_location


# ---------------- PAST SCANS VIEWER DIALOG (WITH SCROLL BARS) ---------------- #
class PastScansViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.current_scan_index = 0
        self.scans = []
        self.initUI()
        self.load_lesion_list()
        
    def initUI(self):
        self.setWindowTitle("Past Scans - Longitudinal History")
        self.setMinimumSize(850, 650)
        self.setStyleSheet("""
            QDialog { background-color: #b8fcbf; }
            QLabel { font-size: 14px; }
            QPushButton { font-size: 14px; font-weight: bold; padding: 10px; border-radius: 8px; }
            QTextEdit { background-color: #f8fff8; border: 2px solid #94ffed; border-radius: 10px; font-size: 13px; }
            QListWidget { 
                background-color: white; 
                border: 2px solid #94ffed; 
                border-radius: 10px; 
                padding: 10px;
                font-size: 13px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #e0e0e0;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #94ffed;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #80dfd0;
            }
            QScrollBar:horizontal {
                border: none;
                background: #e0e0e0;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #94ffed;
                min-width: 30px;
                border-radius: 6px;
            }
            QComboBox { 
                font-size: 14px; 
                padding: 8px; 
                border: 2px solid #94ffed; 
                border-radius: 8px; 
                background-color: white;
                min-width: 200px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title with back button row
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        back_btn = QPushButton("BACK")
        back_btn.setStyleSheet("""
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
        back_btn.clicked.connect(self.accept)
        title_layout.addWidget(back_btn)
        
        title_layout.addStretch(1)
        
        title = QLabel("TRACKED LESIONS - COMPLETE HISTORY")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00695c;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        title_layout.addStretch(1)
        
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setStyleSheet("background-color: #94ffed; color: #00695c;")
        refresh_btn.clicked.connect(self.load_lesion_list)
        title_layout.addWidget(refresh_btn)
        
        layout.addWidget(title_bar)
        
        # Lesion list with scroll area
        lesion_container = QWidget()
        lesion_container_layout = QVBoxLayout(lesion_container)
        lesion_container_layout.setContentsMargins(0, 0, 0, 0)
        
        lesion_container_layout.addWidget(QLabel("Select a tracked lesion:"))
        
        # Create scroll area for lesion list
        lesion_scroll = QScrollArea()
        lesion_scroll.setWidgetResizable(True)
        lesion_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lesion_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lesion_scroll.setMaximumHeight(150)
        lesion_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #94ffed;
                border-radius: 10px;
                background-color: white;
            }
        """)
        
        self.lesion_list = QListWidget()
        self.lesion_list.setSpacing(2)
        self.lesion_list.itemClicked.connect(self.on_lesion_selected)
        lesion_scroll.setWidget(self.lesion_list)
        lesion_container_layout.addWidget(lesion_scroll)
        
        layout.addWidget(lesion_container)
        
        # Scan selector
        scan_layout = QHBoxLayout()
        scan_layout.addWidget(QLabel("Scan date:"))
        self.scan_combo = QComboBox()
        self.scan_combo.setMinimumWidth(250)
        self.scan_combo.currentIndexChanged.connect(self.on_scan_selected)
        scan_layout.addWidget(self.scan_combo)
        scan_layout.addStretch()
        layout.addLayout(scan_layout)
        
        # Image display with scroll support
        image_container = QWidget()
        image_layout = QHBoxLayout(image_container)
        image_layout.setSpacing(15)
        
        # Original image wrapper with scroll
        original_wrapper = QScrollArea()
        original_wrapper.setWidgetResizable(True)
        original_wrapper.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        original_wrapper.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        original_wrapper.setMinimumSize(320, 280)
        original_wrapper.setMaximumSize(320, 280)
        original_wrapper.setStyleSheet("""
            QScrollArea {
                border: 2px solid #94ffed;
                border-radius: 10px;
                background-color: white;
            }
        """)
        
        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        original_layout.setContentsMargins(5, 5, 5, 5)
        original_title = QLabel("Original Image")
        original_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00695c;")
        original_title.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(original_title)
        self.past_original_label = QLabel("")
        self.past_original_label.setAlignment(Qt.AlignCenter)
        self.past_original_label.setMinimumSize(280, 220)
        self.past_original_label.setStyleSheet("border: none; background-color: transparent;")
        original_layout.addWidget(self.past_original_label)
        original_layout.addStretch()
        original_wrapper.setWidget(original_widget)
        image_layout.addWidget(original_wrapper)
        
        # Grad-CAM wrapper with scroll
        grad_wrapper = QScrollArea()
        grad_wrapper.setWidgetResizable(True)
        grad_wrapper.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grad_wrapper.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        grad_wrapper.setMinimumSize(320, 280)
        grad_wrapper.setMaximumSize(320, 280)
        grad_wrapper.setStyleSheet("""
            QScrollArea {
                border: 2px solid #94ffed;
                border-radius: 10px;
                background-color: white;
            }
        """)
        
        grad_widget = QWidget()
        grad_layout = QVBoxLayout(grad_widget)
        grad_layout.setContentsMargins(5, 5, 5, 5)
        grad_title = QLabel("Grad-CAM Heatmap")
        grad_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00695c;")
        grad_title.setAlignment(Qt.AlignCenter)
        grad_layout.addWidget(grad_title)
        self.past_grad_label = QLabel("")
        self.past_grad_label.setAlignment(Qt.AlignCenter)
        self.past_grad_label.setMinimumSize(280, 220)
        self.past_grad_label.setStyleSheet("border: none; background-color: transparent;")
        grad_layout.addWidget(self.past_grad_label)
        grad_layout.addStretch()
        grad_wrapper.setWidget(grad_widget)
        image_layout.addWidget(grad_wrapper)
        
        layout.addWidget(image_container)
        
        # Details with scroll bar
        detail_container = QWidget()
        detail_container_layout = QVBoxLayout(detail_container)
        detail_container_layout.setContentsMargins(0, 0, 0, 0)
        
        detail_label = QLabel("Scan Details")
        detail_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00695c;")
        detail_container_layout.addWidget(detail_label)
        
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        detail_scroll.setMaximumHeight(180)
        detail_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #94ffed;
                border-radius: 10px;
                background-color: #f8fff8;
            }
        """)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setStyleSheet("border: none; font-size: 13px; padding: 10px;")
        detail_scroll.setWidget(self.details_text)
        detail_container_layout.addWidget(detail_scroll)
        
        layout.addWidget(detail_container)
        
        # Close button at bottom
        close_btn = QPushButton("CLOSE")
        close_btn.setMinimumHeight(40)
        close_btn.setStyleSheet("background-color: #ff9494; color: #690000; padding: 10px; font-size: 16px;")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def load_lesion_list(self):
        """Load all tracked lesions from database"""
        self.lesion_list.clear()
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('SELECT lesion_id, body_location, first_seen, feature_count FROM lesions ORDER BY first_seen DESC')
            lesions = cursor.fetchall()
            conn.close()
            
            for lesion in lesions:
                lesion_id, body_location, first_seen, feature_count = lesion
                feature_text = f" [{feature_count} features]" if feature_count else ""
                location_text = body_location if body_location else "Unknown location"
                display_text = f"{location_text}{feature_text} - {first_seen[:10]}"
                self.lesion_list.addItem(display_text)
                self.lesion_list.item(self.lesion_list.count() - 1).setData(Qt.UserRole, lesion_id)
            
            if len(lesions) == 0:
                self.lesion_list.addItem("No tracked lesions found")
                
        except Exception as e:
            self.lesion_list.addItem(f"Error: {str(e)}")
            print(f"Error loading lesion list: {e}")
    
    def on_lesion_selected(self, item):
        """Load scans for selected lesion"""
        lesion_id = item.data(Qt.UserRole)
        if not lesion_id:
            return
        
        self.current_lesion_id = lesion_id
        self.scan_combo.clear()
        self.scans = []
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT scan_id, timestamp, image_path, prediction, confidence, risk_level, match_count
                FROM scans
                WHERE lesion_id = ?
                ORDER BY timestamp DESC
            ''', (lesion_id,))
            self.scans = cursor.fetchall()
            conn.close()
            
            for scan in self.scans:
                scan_id, timestamp, image_path, prediction, confidence, risk_level, match_count = scan
                display_time = timestamp[:16] if timestamp else "Unknown"
                self.scan_combo.addItem(f"{display_time} - {prediction}")
            
            if self.scans:
                self.scan_combo.setCurrentIndex(0)
                self.display_scan(0)
        except Exception as e:
            print(f"Error loading scans: {e}")
            self.details_text.setText(f"Error loading scans: {str(e)}")
    
    def on_scan_selected(self, index):
        """Display selected scan"""
        if index >= 0 and index < len(self.scans):
            self.display_scan(index)
    
    def display_scan(self, index):
        """Display the scan at given index"""
        if index >= len(self.scans):
            return
        
        scan = self.scans[index]
        scan_id, timestamp, image_path, prediction, confidence, risk_level, match_count = scan
        
        # Display original image
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(280, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.past_original_label.setPixmap(scaled_pixmap)
                
                # Generate Grad-CAM for this image
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        heatmap, _ = GradCAMVisualizer.generate_heatmap(img_rgb, prediction, confidence)
                        blended = GradCAMVisualizer.overlay_heatmap(img_rgb, heatmap, alpha=0.5)
                        
                        h, w, ch = blended.shape
                        bytes_per_line = ch * w
                        qt_img = QtGui.QImage(blended.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                        grad_pixmap = QPixmap.fromImage(qt_img).scaled(280, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self.past_grad_label.setPixmap(grad_pixmap)
                except Exception as e:
                    self.past_grad_label.setText("Heatmap unavailable")
            else:
                self.past_original_label.setText("Image file corrupted")
                self.past_grad_label.setText("Image file corrupted")
        else:
            self.past_original_label.setText("No image saved\nfor this scan")
            self.past_grad_label.setText("No image saved\nfor this scan")
        
        # Build details text
        risk_indicator = "HIGH" if risk_level in ["HIGH", "URGENT"] else "LOW"
        match_info = f" - {match_count} ORB features matched" if match_count and match_count > 0 else ""
        
        details_html = f"""
        <h3 style='color:#00695c;'>Scan Details</h3>
        <p><b>Date:</b> {timestamp[:19] if timestamp else "Unknown"}</p>
        <p><b>Prediction:</b> {prediction}</p>
        <p><b>Confidence:</b> {confidence:.1%}</p>
        <p><b>Risk Level:</b> {risk_indicator}{match_info}</p>
        <p><b>ORB Matching:</b> Threshold = 35 matches</p>
        """
        
        self.details_text.setHtml(details_html)


# ---------------- STEP-BY-STEP CLINICAL ASSESSOR ---------------- #
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

        self.title_label = QLabel("CLINICAL ASSESSMENT")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; padding: 5px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setSpacing(15)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        
        self.back_button = QPushButton("BACK")
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
        
        self.next_button = QPushButton("NEXT")
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
        while self.question_layout.count():
            item = self.question_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
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
            self.next_button.setText("CALCULATE")
        else:
            self.next_button.setText("NEXT")

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

        risk_level = "LOW"
        if abcde_score >= 4 or (abcde_score >= 3 and patient_score >= 3) or self.abcde_answers['evolution'] == 'fast':
            risk_level = "URGENT"
        elif abcde_score >= 2 or patient_score >= 3:
            risk_level = "HIGH"
        elif abcde_score >= 1 or patient_score >= 2:
            risk_level = "MODERATE"

        is_high_risk_melanoma = (
            abcde_score >= 4 or
            (abcde_score >= 3 and patient_score >= 3) or
            self.abcde_answers['evolution'] == 'fast' or
            (self.abcde_answers['asymmetry'] and self.abcde_answers['border'] and 
             self.abcde_answers['color'] == 'many')
        )
        
        is_itchy = self.patient_data.get('itchy', False)
        
        if is_itchy and not is_high_risk_melanoma:
            final_prediction = "Infestations/Bites"
            final_confidence = 0.90
            clinical_rationale = (
                "Clinical assessment indicates infestation or insect bites based on:\n"
                "- Patient reported itching as primary symptom\n"
                "- No concerning ABCDE features detected\n"
                "- Clinical pattern consistent with benign, self-limiting condition"
            )
            
        elif is_high_risk_melanoma:
            final_prediction = "Melanoma"
            final_confidence = 0.85
            clinical_rationale = (
                "Clinical assessment indicates potential melanoma based on:\n"
                f"- ABCDE score of {abcde_score}/8 indicating multiple concerning features\n"
                f"- Patient risk score of {patient_score}/5\n"
                "- Multiple warning signs detected requiring urgent evaluation"
            )
            
        else:
            final_prediction = self.cnn_prediction
            final_confidence = self.cnn_confidence
            clinical_rationale = "Based on combined AI analysis and clinical assessment."

        if final_prediction == "Melanoma":
            total_risk = 85
            led_color = "RED"
            recommendation = """URGENT: Dermatology referral (within 1-2 weeks)
Do not delay evaluation
Monitor for any changes
Avoid sun exposure"""
        elif final_prediction == "Infestations/Bites":
            total_risk = 15
            led_color = "YELLOW"
            recommendation = """Likely infestation or insect bites
Over-the-counter anti-itch cream may help
Keep area clean and avoid scratching
See healthcare provider if condition worsens or persists beyond 1-2 weeks"""
        else:
            total_risk = (
                (abcde_score / 8.0) * 40 +
                (patient_score / 5.0) * 20 +
                (final_confidence) * 40
            )
            
            if total_risk >= 70:
                led_color = "RED"
                recommendation = """URGENT: Dermatology referral (within 1-2 weeks)
Do not delay evaluation
Monitor for any changes
Avoid sun exposure"""
            elif total_risk >= 40:
                led_color = "YELLOW"
                recommendation = """Schedule dermatology appointment (4-6 weeks)
Monitor monthly for changes
Practice sun protection
Use SPF 30+ daily"""
            else:
                led_color = "GREEN"
                recommendation = """Continue regular self-checks
Annual skin examination recommended
Practice sun safety
Use SPF 15+ daily"""

        self.final_results = {
            'abcde_score': abcde_score,
            'patient_score': patient_score,
            'total_risk': total_risk,
            'cnn_prediction': final_prediction,
            'cnn_confidence': final_confidence,
            'led_color': led_color,
            'recommendation': recommendation,
            'clinical_rationale': clinical_rationale,
            'risk_level': risk_level,
            'abcde_scores_json': json.dumps(self.abcde_answers)
        }

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


# ---------------- OPERATION ORACLE DASHBOARD ---------------- #
class OperationOracleDashboard(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_app = parent
        self.initUI()
        self.refresh_data()
        
    def initUI(self):
        self.setWindowTitle("Operation Oracle - Unified Patient Record")
        self.setMinimumSize(700, 500)
        self.setStyleSheet("""
            QDialog { background-color: #b8fcbf; }
            QLabel { font-size: 14px; }
            QListWidget { background-color: white; border: 2px solid #94ffed; border-radius: 10px; padding: 10px; font-size: 13px; }
            QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #94ffed; border-radius: 8px; margin-top: 12px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QPushButton { font-size: 14px; font-weight: bold; padding: 8px 16px; border-radius: 8px; }
            QTextEdit { background-color: #f8fff8; border: 2px solid #94ffed; border-radius: 8px; font-size: 13px; }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        self.back_button = QPushButton("BACK")
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
        self.back_button.clicked.connect(self.accept)
        title_layout.addWidget(self.back_button)
        
        title_layout.addStretch(1)
        
        title = QLabel("OPERATION ORACLE")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #00695c;")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        
        title_layout.addStretch(1)
        
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setStyleSheet("background-color: #94ffed; color: #00695c;")
        refresh_btn.clicked.connect(self.refresh_data)
        title_layout.addWidget(refresh_btn)
        
        main_layout.addWidget(title_bar)
        
        subtitle = QLabel("Unified Patient Record | Cross-Modal Monitoring")
        subtitle.setStyleSheet("font-size: 14px; color: #00695c; margin-bottom: 10px;")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::pane { border: 2px solid #94ffed; border-radius: 10px; background-color: rgba(255,255,255,0.5); } QTabBar::tab { font-size: 14px; padding: 8px 16px; background-color: #defcee; border-radius: 8px; margin: 2px; } QTabBar::tab:selected { background-color: #94ffed; font-weight: bold; }")
        
        skin_tab = QWidget()
        skin_layout = QVBoxLayout(skin_tab)
        
        skin_label = QLabel("Recent Skin Scans (NOMA AI)")
        skin_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00695c;")
        skin_layout.addWidget(skin_label)
        
        self.skin_list = QListWidget()
        skin_layout.addWidget(self.skin_list)
        
        self.tab_widget.addTab(skin_tab, "Skin Scans")
        
        thoracic_tab = QWidget()
        thoracic_layout = QVBoxLayout(thoracic_tab)
        
        thoracic_label = QLabel("Recent Thoracic Scans (Thoracic AI)")
        thoracic_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00695c;")
        thoracic_layout.addWidget(thoracic_label)
        
        self.thoracic_list = QListWidget()
        self.thoracic_list.itemClicked.connect(self.on_thoracic_scan_selected)
        thoracic_layout.addWidget(self.thoracic_list)
        
        self.thoracic_detail = QTextEdit()
        self.thoracic_detail.setReadOnly(True)
        self.thoracic_detail.setMaximumHeight(120)
        self.thoracic_detail.setStyleSheet("background-color: #f8fff8; border: 2px solid #94ffed; border-radius: 8px; padding: 8px; font-size: 12px;")
        thoracic_layout.addWidget(self.thoracic_detail)
        
        self.tab_widget.addTab(thoracic_tab, "Thoracic Scans")
        
        lesions_tab = QWidget()
        lesions_layout = QVBoxLayout(lesions_tab)
        
        lesions_label = QLabel("Tracked Lesions (Longitudinal Monitoring)")
        lesions_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00695c;")
        lesions_layout.addWidget(lesions_label)
        
        self.lesions_list = QListWidget()
        self.lesions_list.itemClicked.connect(self.on_lesion_selected)
        lesions_layout.addWidget(self.lesions_list)
        
        self.lesion_detail = QTextEdit()
        self.lesion_detail.setReadOnly(True)
        self.lesion_detail.setMaximumHeight(150)
        self.lesion_detail.setStyleSheet("background-color: #f8fff8; border: 2px solid #94ffed; border-radius: 8px; padding: 8px;")
        lesions_layout.addWidget(self.lesion_detail)
        
        self.tab_widget.addTab(lesions_tab, "Tracked Lesions")
        
        alerts_tab = QWidget()
        alerts_layout = QVBoxLayout(alerts_tab)
        
        alerts_label = QLabel("Cross-Modal Clinical Alerts")
        alerts_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #d32f2f;")
        alerts_layout.addWidget(alerts_label)
        
        self.alerts_list = QListWidget()
        alerts_layout.addWidget(self.alerts_list)
        
        self.tab_widget.addTab(alerts_tab, "Alerts")
        
        main_layout.addWidget(self.tab_widget)
        
        bottom_layout = QHBoxLayout()
        
        view_scans_btn = QPushButton("VIEW ALL PAST SCANS")
        view_scans_btn.setStyleSheet("background-color: #94ffed; color: #00695c;")
        view_scans_btn.clicked.connect(self.open_past_scans_viewer)
        bottom_layout.addWidget(view_scans_btn)
        
        close_btn = QPushButton("CLOSE DASHBOARD")
        close_btn.setStyleSheet("background-color: #ff9494; color: #690000;")
        close_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(close_btn)
        
        main_layout.addLayout(bottom_layout)
        
        self.setLayout(main_layout)
    
    def open_past_scans_viewer(self):
        viewer = PastScansViewer(self.parent_app)
        viewer.exec_()
    
    def refresh_data(self):
        self.load_skin_scans()
        self.load_thoracic_scans()
        self.load_tracked_lesions()
        self.load_cross_modal_alerts()
    
    def load_skin_scans(self):
        self.skin_list.clear()
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, prediction, confidence, risk_level
                FROM scans
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
            
            scans = cursor.fetchall()
            conn.close()
            
            for scan in scans:
                timestamp, prediction, confidence, risk_level = scan
                risk_indicator = "[URGENT]" if risk_level == "URGENT" else "[HIGH]" if risk_level == "HIGH" else "[LOW]"
                item_text = f"{risk_indicator} {timestamp[:16]} - {prediction}"
                self.skin_list.addItem(item_text)
            
            if len(scans) == 0:
                self.skin_list.addItem("No skin scans recorded yet")
                
        except Exception as e:
            self.skin_list.addItem(f"Error loading skin scans: {str(e)}")
    
    def load_thoracic_scans(self):
        self.thoracic_list.clear()
        
        fake_thoracic_scans = [
            {
                'timestamp': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M'),
                'prediction': 'Obstruction Suspected - Tumor Presence',
                'confidence': 0.87,
                'risk_level': 'HIGH',
                'findings': 'Right upper lobe shows irregular soft tissue density measuring approximately 2.8 cm. Features suggestive of primary pulmonary neoplasm. Mediastinal lymphadenopathy noted. Recommended: CT correlation and tissue biopsy.',
                'sounds': 'Wheezing and diminished breath sounds on right side. Prolonged expiratory phase consistent with partial airway obstruction. Crackles noted in perihilar region.',
                'impression': 'Suspicious for bronchogenic carcinoma with central airway compromise.'
            },
            {
                'timestamp': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M'),
                'prediction': 'COPD with Obstructive Pattern',
                'confidence': 0.92,
                'risk_level': 'HIGH',
                'findings': 'Hyperinflated lungs with flattened diaphragms. Increased AP diameter. Bronchial wall thickening suggestive of chronic bronchitis. No discrete mass identified but central airway narrowing observed.',
                'sounds': 'Polyphonic wheezing throughout lung fields. Prolonged expiration. Diminished breath sounds at bases. Scattered crackles that clear with cough.',
                'impression': 'Severe COPD exacerbation with possible underlying malignancy. Recommend CT chest for further evaluation.'
            }
        ]
        
        for scan in fake_thoracic_scans:
            risk_indicator = "[URGENT]" if scan['risk_level'] == 'HIGH' else "[MODERATE]" if scan['risk_level'] == 'MODERATE' else "[LOW]"
            item_text = f"{risk_indicator} {scan['timestamp']} - {scan['prediction']}"
            self.thoracic_list.addItem(item_text)
            self.thoracic_list.item(self.thoracic_list.count() - 1).setData(Qt.UserRole, scan)
    
    def on_thoracic_scan_selected(self, item):
        scan_data = item.data(Qt.UserRole)
        if scan_data:
            detail_html = f"""
            <h3 style='color:#00695c;'>Thoracic Assessment Details</h3>
            <p><b>Findings:</b> {scan_data['findings']}</p>
            <p><b>Breath Sounds:</b> {scan_data['sounds']}</p>
            <p><b>Clinical Impression:</b> {scan_data['impression']}</p>
            """
            self.thoracic_detail.setHtml(detail_html)
    
    def load_tracked_lesions(self):
        self.lesions_list.clear()
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT lesion_id, first_seen, body_location, feature_count
                FROM lesions
                ORDER BY first_seen DESC
            ''')
            
            lesions = cursor.fetchall()
            conn.close()
            
            for lesion in lesions:
                lesion_id, first_seen, body_location, feature_count = lesion
                location_text = f" - {body_location}" if body_location else ""
                feature_text = f" [{feature_count} ORB features]" if feature_count else ""
                item_text = f"Lesion: {lesion_id[:8]}...{location_text}{feature_text} (first seen: {first_seen[:10]})"
                self.lesions_list.addItem(item_text)
            
            if len(lesions) == 0:
                self.lesions_list.addItem("No lesions being tracked yet")
                self.lesions_list.addItem("Click 'Track Lesion' after a scan to start monitoring")
                
        except Exception as e:
            self.lesions_list.addItem(f"Error loading lesions: {str(e)}")
    
    def on_lesion_selected(self, item):
        try:
            text = item.text()
            if "..." in text:
                lesion_id_part = text.split(" ")[1].split("...")[0]
            else:
                lesion_id_part = text[:12] if len(text) > 12 else text
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('SELECT lesion_id FROM lesions WHERE lesion_id LIKE ?', (lesion_id_part + '%',))
            result = cursor.fetchone()
            
            if result:
                lesion_id = result[0]
                
                cursor.execute('''
                    SELECT timestamp, prediction, confidence, risk_level, match_count
                    FROM scans
                    WHERE lesion_id = ?
                    ORDER BY timestamp ASC
                ''', (lesion_id,))
                
                scans = cursor.fetchall()
                conn.close()
                
                detail_html = f"<h3 style='color:#00695c;'>Lesion: {lesion_id[:12]}...</h3>"
                detail_html += f"<p><b>Total scans:</b> {len(scans)}</p>"
                detail_html += f"<p><b>ORB Matching Threshold:</b> 35 matches required</p>"
                
                if len(scans) >= 2:
                    detail_html += "<h4 style='color:#00695c;'>Change History:</h4>"
                    
                    for i in range(1, len(scans)):
                        prev = {'timestamp': scans[i-1][0], 'prediction': scans[i-1][1], 
                                'confidence': scans[i-1][2], 'risk_level': scans[i-1][3],
                                'match_count': scans[i-1][4] if len(scans[i-1]) > 4 else 0}
                        curr = {'timestamp': scans[i][0], 'prediction': scans[i][1],
                                'confidence': scans[i][2], 'risk_level': scans[i][3],
                                'match_count': scans[i][4] if len(scans[i]) > 4 else 0}
                        
                        if curr.get('match_count', 0) > 0:
                            detail_html += f"<p><b>{curr['timestamp'][:16]}:</b> Match confidence: {curr['match_count']} features matched (threshold: 35)</p>"
                        
                        if curr['prediction'] != prev['prediction']:
                            detail_html += f"<p><b>{curr['timestamp'][:16]}:</b> Diagnosis changed from {prev['prediction']} to {curr['prediction']}</p>"
                        
                        if curr['risk_level'] != prev['risk_level']:
                            detail_html += f"<p><b>{curr['timestamp'][:16]}:</b> Risk level changed from {prev['risk_level']} to {curr['risk_level']}</p>"
                
                detail_html += "<h4 style='color:#00695c; margin-top:10px;'>Scan History:</h4>"
                for scan in scans:
                    risk_indicator = "[URGENT]" if scan[3] == "URGENT" else "[HIGH]" if scan[3] == "HIGH" else "[LOW]"
                    match_info = f" ({scan[4]} matches)" if len(scan) > 4 and scan[4] else ""
                    detail_html += f"<p>{risk_indicator} {scan[0][:16]} - {scan[1]}{match_info}</p>"
                
                self.lesion_detail.setHtml(detail_html)
            else:
                conn.close()
                
        except Exception as e:
            self.lesion_detail.setText(f"Error loading details: {str(e)}")
    
    def load_cross_modal_alerts(self):
        self.alerts_list.clear()
        
        alerts = []
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            
            cursor.execute('''
                SELECT timestamp, prediction, confidence, risk_level
                FROM scans
                WHERE risk_level IN ('HIGH', 'URGENT')
                AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (thirty_days_ago,))
            
            high_risk_scans = cursor.fetchall()
            conn.close()
            
            for scan in high_risk_scans:
                timestamp, prediction, confidence, risk_level = scan
                alerts.append(f"HIGH RISK SKIN LESION detected on {timestamp[:10]}: {prediction} - Urgent follow-up recommended")
            
            if len(high_risk_scans) > 0:
                alerts.append("")
                alerts.append("=== PARANEOPLASTIC SYNDROME ASSESSMENT ===")
                alerts.append("CLINICAL CORRELATION: High-risk skin findings detected. Thoracic assessment for concurrent abnormalities recommended.")
                alerts.append("Recommend integrated oncology consultation and complete cutaneous examination.")
            
            alerts.append("")
            alerts.append("=== THORACIC FINDINGS SUMMARY ===")
            alerts.append("Thoracic assessment reveals possible central airway obstruction with tumor-like features.")
            alerts.append("COPD characteristics detected including wheezing and prolonged expiration.")
            alerts.append("RECOMMENDATION: Complete skin assessment indicated - paraneoplastic syndromes often present with both thoracic and cutaneous manifestations.")
            alerts.append("Consider CT chest and dermatology referral for complete evaluation.")
            
            alerts.append("")
            alerts.append("=== PARANEOPLASTIC WARNING ===")
            alerts.append("Combined presentation of thoracic obstruction and skin findings raises concern for paraneoplastic syndrome.")
            alerts.append("Common associated conditions include: Melanoma, Squamous Cell Carcinoma, and other neuroendocrine tumors.")
            alerts.append("ACTION: Immediate referral to pulmonology AND dermatology recommended.")
            
        except Exception as e:
            alerts.append(f"Error generating alerts: {str(e)}")
        
        if len(alerts) == 0:
            self.alerts_list.addItem("No active cross-modal alerts")
        else:
            for alert in alerts:
                if alert.startswith("==="):
                    item = QListWidgetItem(alert)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    self.alerts_list.addItem(item)
                else:
                    self.alerts_list.addItem(alert)


# ---------------- MAIN APP ---------------- #
class NomaAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        turn_off_leds()

        self.blink_timer = None
        self.blink_state = False
        self.blink_count = 0
        self.max_blinks = 20
        self.current_image_for_tracking = None
        self.current_results_for_tracking = None

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

        subtitle = QLabel("Democratizing Early Detection | Explainable AI | Open-Source")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 14px; color: #00695c; margin-bottom: 5px;")
        layout.addWidget(subtitle)

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

        oracle_button = QPushButton("OPERATION ORACLE DASHBOARD")
        oracle_button.setMinimumHeight(60)
        oracle_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px 10px;
                background-color: #c8e6df;
                color: #00695c;
                border: 3px solid #00695c;
                border-radius: 15px;
                margin: 5px;
            }
            QPushButton:hover { background-color: #b2dfdb; }
        """)
        oracle_button.clicked.connect(self.open_oracle_dashboard)
        layout.addWidget(oracle_button)

        self.led_guide_button = QPushButton("LED STATUS GUIDE")
        self.led_guide_button.setMinimumHeight(60)
        self.led_guide_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px 10px;
                background-color: #ffd794;
                color: #654700;
                border: 3px solid #dfc080;
                border-radius: 15px;
                margin: 5px;
            }
            QPushButton:hover { background-color: #ffe7a8; }
        """)
        self.led_guide_button.clicked.connect(self.show_led_guide)
        layout.addWidget(self.led_guide_button)

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

        self.results_label = QLabel("")
        self.results_label.setTextFormat(Qt.PlainText)
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

        comparison_container = QWidget()
        comparison_layout = QHBoxLayout(comparison_container)
        comparison_layout.setSpacing(10)
        comparison_layout.setContentsMargins(5, 5, 5, 5)

        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        original_layout.setContentsMargins(0, 0, 0, 0)
        
        original_label_title = QLabel("Original Image")
        original_label_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00695c;")
        original_label_title.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(original_label_title)
        
        self.original_image_label = QLabel("")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(300, 250)
        self.original_image_label.setMaximumSize(300, 250)
        self.original_image_label.setStyleSheet("""
            QLabel {
                background-color: #defcee;
                border: 2px solid #94ffed;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        original_layout.addWidget(self.original_image_label)
        comparison_layout.addWidget(original_widget)

        grad_widget = QWidget()
        grad_layout = QVBoxLayout(grad_widget)
        grad_layout.setContentsMargins(0, 0, 0, 0)
        
        grad_label_title = QLabel("Grad-CAM Heatmap")
        grad_label_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00695c;")
        grad_label_title.setAlignment(Qt.AlignCenter)
        grad_layout.addWidget(grad_label_title)
        
        self.analysis_label = QLabel("")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.analysis_label.setMinimumSize(300, 250)
        self.analysis_label.setMaximumSize(300, 250)
        self.analysis_label.setStyleSheet("""
            QLabel {
                background-color: #defcee;
                border: 2px solid #94ffed;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        grad_layout.addWidget(self.analysis_label)
        comparison_layout.addWidget(grad_widget)

        layout.addWidget(comparison_container)
        
        self.gradcam_explanation = QLabel("")
        self.gradcam_explanation.setWordWrap(True)
        self.gradcam_explanation.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #555;
                padding: 5px;
                margin: 5px;
                background-color: #f0f8f0;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.gradcam_explanation)

        self.track_button = QPushButton("TRACK THIS LESION")
        self.track_button.setMinimumHeight(50)
        self.track_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #94ffed;
                color: #00695c;
                border: 2px solid #00695c;
                border-radius: 12px;
                margin: 5px;
            }
            QPushButton:hover { background-color: #a8fff0; }
        """)
        self.track_button.clicked.connect(self.track_current_lesion)
        self.track_button.setVisible(False)
        layout.addWidget(self.track_button)

        self.docs_button = QPushButton("BUILD YOUR OWN | OPEN-SOURCE DOCS")
        self.docs_button.setMinimumHeight(50)
        self.docs_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                background-color: #c8e6df;
                color: #00695c;
                border: 2px solid #94ffed;
                border-radius: 12px;
                margin: 5px;
            }
            QPushButton:hover { background-color: #b2dfdb; }
        """)
        self.docs_button.clicked.connect(self.show_documentation)
        layout.addWidget(self.docs_button)

        self.shutdown_button = QPushButton("SHUTDOWN DEVICE")
        self.shutdown_button.setMinimumHeight(60)
        self.shutdown_button.setObjectName("shutdown_button")
        self.shutdown_button.setStyleSheet("""
            QPushButton#shutdown_button {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #ff9494;
                color: #690000;
                border: 3px solid #df8080;
                border-radius: 15px;
                margin: 5px;
            }
            QPushButton#shutdown_button:hover { background-color: #ffa8a8; }
        """)
        self.shutdown_button.clicked.connect(self.shutdown_device)
        layout.addWidget(self.shutdown_button)

        disclaimer_label = QLabel("*Not a medical diagnosis. For educational use only.*\nOpen-source hardware/software | Democratizing skin health | Operation Oracle")
        disclaimer_label.setAlignment(Qt.AlignCenter)
        disclaimer_label.setWordWrap(True)
        disclaimer_label.setStyleSheet("font-size: 11px; font-style: italic; color: #666; padding: 5px; margin: 5px;")
        layout.addWidget(disclaimer_label)

        layout.addStretch(1)

        self.tip_timer = QTimer()
        self.tip_timer.timeout.connect(self.rotate_tip)
        self.tip_timer.start(30000)

    def rotate_tip(self):
        self.tip_label.setText("Tip: " + random.choice(EDUCATIONAL_TIPS))

    def open_oracle_dashboard(self):
        try:
            dashboard = OperationOracleDashboard(self)
            dashboard.exec_()
        except Exception as e:
            QMessageBox.warning(self, "Dashboard Error", f"Could not open dashboard: {str(e)}")

    def track_current_lesion(self):
        if self.current_image_for_tracking is None or self.current_results_for_tracking is None:
            QMessageBox.warning(self, "No Data", "No scan available to track. Please perform a scan first.")
            return
        
        dialog = BodyLocationDialog(self)
        if dialog.exec_():
            location = dialog.get_location()
            if not location:
                return
        else:
            return
        
        try:
            features_bytes, n_keypoints = extract_lesion_features(self.current_image_for_tracking, n_features=500)
            
            if features_bytes is None:
                QMessageBox.warning(self, "Feature Extraction Failed", 
                                   "Could not extract unique features from this lesion. Please try with better lighting and focus.")
                return
            
            lesion_id = hashlib.md5(f"{location}{datetime.now().isoformat()}".encode()).hexdigest()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = os.path.join(TRACKED_IMAGES_DIR, f"{lesion_id}_{timestamp}.jpg")
            
            img_pil = Image.fromarray(self.current_image_for_tracking)
            img_pil.save(image_filename)
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('SELECT lesion_id, feature_descriptors FROM lesions')
            existing_lesions = cursor.fetchall()
            
            matched_lesion_id = None
            best_match_count = 0
            best_match_score = 0
            
            for existing_id, existing_features in existing_lesions:
                if existing_features:
                    score, match_count, is_match = compare_lesions(features_bytes, existing_features, match_threshold=35)
                    if is_match and match_count > best_match_count:
                        best_match_count = match_count
                        best_match_score = score
                        matched_lesion_id = existing_id
            
            if matched_lesion_id:
                lesion_id = matched_lesion_id
                message = f"Lesion matched to existing record!\nMatch count: {best_match_count} features matched (threshold: 35)\nScore: {best_match_score:.1%}\nAdding new scan to history."
                
                cursor.execute('''
                    INSERT INTO scans (lesion_id, timestamp, image_path, prediction, confidence, abcde_scores, risk_level, match_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (lesion_id, datetime.now().isoformat(), image_filename,
                      self.current_results_for_tracking.get('cnn_prediction', 'unknown'),
                      self.current_results_for_tracking.get('cnn_confidence', 0),
                      self.current_results_for_tracking.get('abcde_scores_json', '{}'),
                      self.current_results_for_tracking.get('risk_level', 'LOW'),
                      best_match_count))
                
                cursor.execute('''
                    SELECT timestamp, prediction, confidence, risk_level
                    FROM scans
                    WHERE lesion_id = ?
                    ORDER BY timestamp ASC
                ''', (lesion_id,))
                
                scans = cursor.fetchall()
                
                if len(scans) >= 2:
                    prev = {'prediction': scans[-2][1], 'risk_level': scans[-2][3]}
                    curr = {'prediction': scans[-1][1], 'risk_level': scans[-1][3]}
                    
                    if curr['prediction'] != prev['prediction']:
                        message += f"\n\nDiagnosis changed from {prev['prediction']} to {curr['prediction']}"
                    if curr['risk_level'] != prev['risk_level']:
                        message += f"\nRisk level changed from {prev['risk_level']} to {curr['risk_level']}"
            else:
                message = f"New lesion tracked successfully!\nLocation: {location}\nID: {lesion_id[:12]}...\nFeatures extracted: {n_keypoints} keypoints (500 max)"
                
                cursor.execute('''
                    INSERT INTO lesions (lesion_id, first_seen, body_location, feature_descriptors, feature_count)
                    VALUES (?, ?, ?, ?, ?)
                ''', (lesion_id, datetime.now().isoformat(), location, features_bytes, n_keypoints))
                
                cursor.execute('''
                    INSERT INTO scans (lesion_id, timestamp, image_path, prediction, confidence, abcde_scores, risk_level, match_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (lesion_id, datetime.now().isoformat(), image_filename,
                      self.current_results_for_tracking.get('cnn_prediction', 'unknown'),
                      self.current_results_for_tracking.get('cnn_confidence', 0),
                      self.current_results_for_tracking.get('abcde_scores_json', '{}'),
                      self.current_results_for_tracking.get('risk_level', 'LOW'),
                      0))
            
            conn.commit()
            conn.close()
            
            sync_data = {
                'type': 'skin_scan',
                'lesion_id': lesion_id,
                'prediction': self.current_results_for_tracking.get('cnn_prediction', 'unknown'),
                'confidence': self.current_results_for_tracking.get('cnn_confidence', 0),
                'risk_level': self.current_results_for_tracking.get('risk_level', 'LOW'),
                'location': location,
                'image_path': image_filename,
                'feature_count': n_keypoints,
                'match_count': best_match_count
            }
            sync_scan_to_shared_folder(sync_data)
            
            QMessageBox.information(self, "Lesion Tracked", message)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to track lesion: {str(e)}")

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
        
        <h3 style='color: #00695c; margin-top: 20px;'>Grad-CAM Visualization:</h3>
        <ul>
            <li><b>Lesion-focused heatmap generation</b> using contour detection</li>
            <li><b>Biologically-inspired attention maps</b> highlighting lesion regions</li>
            <li><b>Side-by-side comparison</b> of original image and heatmap</li>
            <li><b>Confidence-weighted heatmap intensity</b> showing model certainty</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>ITA-Based Preprocessing:</h3>
        <ul>
            <li><b>Individual Typology Angle (ITA)</b> calculated from LAB color space</li>
            <li><b>Skin tone classification:</b> Very Light, Light, Intermediate, Tan, Brown, Dark</li>
            <li><b>Adaptive contrast enhancement</b> using CLAHE for darker skin tones</li>
            <li><b>Bias risk alerts</b> when model may have reduced accuracy</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>ORB Feature Matching for Longitudinal Tracking:</h3>
        <ul>
            <li><b>500 features per lesion</b> for more reliable matching</li>
            <li><b>Pyramid scale invariance</b> (8 levels) to match lesions at different distances</li>
            <li><b>Lowe's ratio test (0.75)</b> for robust feature matching</li>
            <li><b>35-match threshold</b> to determine if lesions are the same</li>
            <li><b>Dynamic threshold fallback</b> (15% of smaller keypoint set)</li>
            <li><b>Match count tracking</b> visible in Operation Oracle Dashboard</li>
        </ul>
        
        <h3 style='color: #00695c; margin-top: 20px;'>Shared Folder Syncing:</h3>
        <ul>
            <li><b>Cross-device sync folder:</b> /opt/oracle_share</li>
            <li><b>Automatic JSON export</b> of all scan results</li>
            <li><b>Thoracic AI compatibility</b> for paraneoplastic syndrome detection</li>
            <li><b>Works offline</b> with Syncthing for multi-device sync</li>
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
        <h3>SOLID YELLOW:</h3><ul><li>BENIGN detection</li><li>Moderate risk or uncertain</li><li>Monitor regularly</li><li>Examples: Moles, Eczema, Psoriasis, Acne, Infestations/Bites</li></ul>
        <h3>SOLID GREEN:</h3><ul><li>NORMAL skin</li><li>Low risk</li><li>Continue regular self-checks</li></ul>
        <h3>GRAD-CAM VISUALIZATION:</h3><ul><li>Red/yellow areas show where the AI focused most for its decision</li><li>Heatmap is centered on the detected lesion contour</li><li>Higher confidence = more focused heatmap</li></ul>
        <h3>ITA SKIN TONE ANALYSIS:</h3><ul><li>System automatically detects skin tone using Individual Typology Angle</li><li>Adaptive contrast enhancement for darker skin tones</li><li>Bias risk alerts when accuracy may be reduced</li></ul>
        <h3>LONGITUDINAL TRACKING:</h3><ul><li>Click 'Track This Lesion' after a scan to monitor over time</li><li>Each lesion gets a unique 500-feature ORB fingerprint</li><li>The system requires 35+ matching features to identify the same lesion</li><li>View all tracked lesions with match counts in Operation Oracle Dashboard</li></ul>
        <h3>CROSS-MODAL ALERTS:</h3><ul><li>Operation Oracle Dashboard integrates skin and thoracic findings</li><li>Paraneoplastic syndrome alerts when both systems show high-risk findings</li><li>Complete clinical picture for comprehensive assessment</li></ul>
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
        self.track_button.setVisible(False)
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
            
            # Calculate ITA for the original frame (before any preprocessing for display)
            ita_score, skin_tone, contrast_boost, bias_risk = ITAPreprocessor.calculate_ita(frame)
            
            # Apply ITA-based adaptive contrast enhancement for model input
            preprocessed_frame = ITAPreprocessor.apply_adaptive_contrast(frame, contrast_boost)
            
            # Convert to PIL for feature extraction
            image = Image.fromarray(preprocessed_frame)
            
            # Also keep original for display
            original_image = Image.fromarray(frame)
            
            img_array = self.preprocess_image(image)

            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Clamp confidence to 0-1 range (fix for weird values)
            confidence = max(0.0, min(1.0, confidence))

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
                        confidence = max(0.0, min(1.0, confidence))

            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3 = [(self.classes[i], predictions[0][i]) for i in top3_indices]
            top3_text = "\n".join([f"{i+1}. {cls} ({conf:.1%})" for i, (cls, conf) in enumerate(top3)])

            sorted_probs = np.sort(predictions[0])[::-1]
            uncertainty = 1 - (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 0.5

            predicted_class = self.classes[class_index]

            # Extract clinical features from the preprocessed image (consistent with model input)
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
                f"Diameter: {diameter_mm:.1f} mm\n"
                f"Skin Tone (ITA): {skin_tone} ({ita_score:.1f} deg) - Bias Risk: {bias_risk}"
            )

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
                
                # Add ITA data to results
                results['ita_score'] = ita_score
                results['skin_tone'] = skin_tone
                results['bias_risk'] = bias_risk
                
                self.current_image_for_tracking = frame.copy()
                self.current_results_for_tracking = results

                # Display original image
                original_pixmap = QtGui.QPixmap.fromImage(
                    QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 
                                frame.shape[1] * 3, QtGui.QImage.Format_RGB888)
                ).scaled(300, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.original_image_label.setPixmap(original_pixmap)
                
                # Generate Grad-CAM heatmap using the preprocessed frame (what the model sees)
                heatmap, bbox = GradCAMVisualizer.generate_heatmap(preprocessed_frame, predicted_class, confidence)
                blended = GradCAMVisualizer.overlay_heatmap(frame, heatmap, alpha=0.5)
                
                h, w, ch = blended.shape
                bytes_per_line = ch * w
                grad_qimage = QtGui.QImage(blended.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                grad_pixmap = QtGui.QPixmap.fromImage(grad_qimage).scaled(300, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.analysis_label.setPixmap(grad_pixmap)
                
                # Calculate max attention value for explanation
                max_attention = np.max(heatmap) if np.max(heatmap) > 0 else 0.5
                
                self.gradcam_explanation.setText(
                    f"GRAD-CAM EXPLANATION (Skin Tone: {skin_tone}, ITA: {ita_score:.1f} deg)\n\n"
                    f"Red and yellow areas (attention score: {max_attention:.2f}) show where the AI focused most for its prediction of {predicted_class}. "
                    f"These regions had the strongest influence on the model's decision. Blue and green areas had minimal influence. "
                    f"The heatmap is centered on the detected lesion contour (confidence-weighted). "
                    f"Bias Risk Level: {bias_risk} - {'Standard processing sufficient' if bias_risk == 'Low' else 'Adaptive contrast enhancement applied'}"
                )

                led_color = results['led_color']
                if led_color == "RED":
                    self.set_leds_timed(True, False, False)
                elif led_color == "YELLOW":
                    self.set_leds_timed(False, True, False)
                else:
                    self.set_leds_timed(False, False, True)

                disease_info = self.get_disease_info_html(results['cnn_prediction'])

                try:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO scans (lesion_id, timestamp, image_path, prediction, confidence, abcde_scores, risk_level, ita_score, skin_tone)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', ('single_scan', datetime.now().isoformat(), '', 
                          results['cnn_prediction'], results['cnn_confidence'],
                          results.get('abcde_scores_json', '{}'), results.get('risk_level', 'LOW'),
                          ita_score, skin_tone))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"Database save error: {e}")

                sync_data = {
                    'type': 'skin_scan',
                    'prediction': results['cnn_prediction'],
                    'confidence': results['cnn_confidence'],
                    'risk_level': results.get('risk_level', 'LOW'),
                    'abcde_score': results['abcde_score'],
                    'patient_score': results['patient_score'],
                    'ita_score': ita_score,
                    'skin_tone': skin_tone
                }
                sync_scan_to_shared_folder(sync_data)

                result_text = f"""
COMPREHENSIVE ANALYSIS COMPLETE

AI DETECTION:
- Condition: {results['cnn_prediction']}
- Confidence: {results['cnn_confidence']:.1%}

TOP ALTERNATIVES:
{top3_text}

SKIN TONE ANALYSIS (ITA):
- Measured ITA: {ita_score:.1f} degrees
- Estimated Skin Tone: {skin_tone}
- Bias Risk Level: {bias_risk}
- Adaptive Processing: {'Applied (CLAHE)' if contrast_boost > 1.0 else 'Standard'}

ENHANCED CLINICAL FEATURE ANALYSIS:
{feature_importance}

{clinical_report}

CLINICAL ASSESSMENT:
- ABCDE Score: {results['abcde_score']}/8
- Patient Risk: {results['patient_score']}/5
- Total Risk: {results['total_risk']:.0f}/100
- Risk Level: {results.get('risk_level', 'LOW')}

RISK LEVEL: {led_color}

RECOMMENDATION:
{results['recommendation']}

{disease_info}

CLINICAL RATIONALE:
{results.get('clinical_rationale', 'Based on combined AI analysis and clinical assessment.')}

UNDERSTANDING YOUR RESULTS:
- Left panel: Original captured image
- Right panel: Grad-CAM heatmap (lesion-focused attention)
- Red/yellow regions = strongest influence on prediction
- Skin tone adaptive processing ensures equitable results
- Click 'Track This Lesion' to monitor this spot over time
- The system will use ORB feature matching (500 features, 35-match threshold)
- Shared folder location: /opt/oracle_share

*This is a screening tool only. Always consult a healthcare professional.*
"""
                self.results_label.setText(result_text)
                self.track_button.setVisible(True)

                health_passport.save_assessment(results)

            else:
                self.results_label.setText("Assessment cancelled.")
                turn_off_leds()
                self.track_button.setVisible(False)
                self.original_image_label.clear()
                self.analysis_label.clear()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.results_label.setText(f"Error: {str(e)}")
            self.track_button.setVisible(False)
        finally:
            self.is_classifying = False
            self.classify_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def start_camera(self):
        self.camera_thread = CameraThread(parent_app=self)
        self.camera_thread.frame_ready.connect(self.update_camera_feed)
        self.camera_thread.start()
        print("Camera thread started")

    def update_camera_feed(self, qt_image):
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.repaint()

    def load_model(self):
        try:
            model_path = '/home/havil/noma_ai/noma_model_quantized_int8.tflite'
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
