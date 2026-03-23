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
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QPushButton, QApplication,
                             QMessageBox, QProgressBar, QMainWindow, QScrollArea,
                             QWidget, QHBoxLayout, QTextEdit, QRadioButton,
                             QSpinBox, QComboBox, QCheckBox, QGroupBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush
from PIL import Image
import os
from datetime import datetime
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

# ---------------- Camera Permission Fix ---------------- #
os.environ['LIBCAMERA_LOG_LEVELS'] = '0'  # Disable libcamera debug logs
os.environ['LIBCAMERA_IPA'] = 'rpi/vc4'   # Force Raspberry Pi pipeline

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

# ---------------- Simple GPIO Controller ---------------- #
class SimpleLED:
    """Simple GPIO controller that works with both real and mock GPIO"""
    
    def __init__(self):
        self.gpio_available = False
        self.led_pins = {'red': 17, 'yellow': 27, 'green': 22}
        self.led_states = {'red': False, 'yellow': False, 'green': False}
        
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.gpio_available = True
            
            # Set up GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup pins as output
            for pin in self.led_pins.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            print("GPIO initialized successfully")
            
        except (ImportError, RuntimeError, PermissionError) as e:
            print(f"GPIO initialization failed, using mock: {e}")
            self.gpio_available = False
    
    def set_led(self, color, state):
        """Set an LED on or off"""
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
        """Turn all LEDs off"""
        for color in self.led_pins:
            self.set_led(color, False)
    
    def cleanup(self):
        """Cleanup GPIO"""
        if self.gpio_available:
            try:
                print("Cleaning up GPIO...")
                self.all_off()
                self.GPIO.cleanup()
                print("GPIO cleanup complete")
            except Exception as e:
                print(f"Error during GPIO cleanup: {e}")

# Create global LED controller
led_controller = SimpleLED()

# Helper functions
def set_leds(red=False, yellow=False, green=False):
    """Set LED states"""
    led_controller.set_led('red', red)
    led_controller.set_led('yellow', yellow)
    led_controller.set_led('green', green)

def turn_off_leds():
    """Turn all LEDs off"""
    led_controller.all_off()

# ---------------- SWIPEABLE STEP-BY-STEP CLINICAL ASSESSOR ---------------- #
class SwipeableWidget(QtWidgets.QWidget):
    """A widget that detects swipe gestures for navigation without blocking child widgets"""
    
    swipe_left = pyqtSignal()
    swipe_right = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.touch_start_pos = None
        self.swipe_threshold = 100  # minimum distance for swipe
        self.touch_in_progress = False
        self.touch_start_time = None
        self.max_swipe_time = 500  # max milliseconds for a swipe
        
    def touchEvent(self, event):
        touch_points = event.touchPoints()
        if len(touch_points) == 1:
            touch = touch_points[0]
            
            if event.type() == QtCore.QEvent.TouchBegin:
                self.touch_start_pos = touch.pos()
                self.touch_start_time = time.time()
                self.touch_in_progress = True
                event.accept()
                return True
                
            elif event.type() == QtCore.QEvent.TouchEnd and self.touch_in_progress:
                if self.touch_start_pos and self.touch_start_time:
                    end_pos = touch.pos()
                    delta_x = end_pos.x() - self.touch_start_pos.x()
                    delta_time = (time.time() - self.touch_start_time) * 1000  # convert to ms
                    
                    # Only trigger swipe if it's fast enough and far enough
                    if delta_time < self.max_swipe_time and abs(delta_x) > self.swipe_threshold:
                        # Find the widget under the touch point
                        widget_under_touch = self.childAt(end_pos.toPoint())
                        
                        # Only trigger swipe if we're not touching an interactive widget
                        if not self.is_interactive_widget(widget_under_touch):
                            if delta_x > 0:  # right swipe
                                self.swipe_right.emit()
                            else:  # left swipe
                                self.swipe_left.emit()
                            event.accept()
                            self.touch_in_progress = False
                            self.touch_start_pos = None
                            return True
                
                self.touch_in_progress = False
                self.touch_start_pos = None
                event.accept()
                return True
        
        return super().touchEvent(event)
    
    def is_interactive_widget(self, widget):
        """Check if the widget is interactive (should receive touches instead of swipe)"""
        if widget is None:
            return False
        
        # List of widget types that should get touch priority
        interactive_types = (
            QtWidgets.QRadioButton,
            QtWidgets.QCheckBox,
            QtWidgets.QPushButton,
            QtWidgets.QComboBox,
            QtWidgets.QSpinBox,
            QtWidgets.QScrollBar,
            QtWidgets.QSlider
        )
        
        # Check if widget or its parent is interactive
        current = widget
        while current:
            if isinstance(current, interactive_types):
                return True
            current = current.parent()
        
        return False
    
    def event(self, event):
        if event.type() == QtCore.QEvent.TouchBegin or \
           event.type() == QtCore.QEvent.TouchUpdate or \
           event.type() == QtCore.QEvent.TouchEnd:
            return self.touchEvent(event)
        return super().event(event)

class StepByStepClinicalAssessor(QtWidgets.QDialog):
    """Step-by-step clinical assessment wizard with progress bar and swipe navigation"""
    
    def __init__(self, parent=None, cnn_prediction="", cnn_confidence=0.0):
        super().__init__(parent)
        self.cnn_prediction = cnn_prediction
        self.cnn_confidence = cnn_confidence
        self.parent_app = parent
        
        # Assessment data storage
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
            'sunburn_history': False
        }
        
        # Wizard state
        self.current_step = 0
        self.total_steps = 7
        
        # Track widgets for cleanup
        self.current_widgets = []
        
        self.initUI()
        
        # Start yellow LED blinking (with 10 second timeout)
        if self.parent_app:
            self.parent_app.start_yellow_blinking_for_dialog()
        
        # Show first step
        self.show_step(0)
    
    def initUI(self):
        self.setWindowTitle("Clinical Assessment Wizard")
        self.setFixedSize(800, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #b8fcbf;
            }
            QLabel {
                font-size: 16px;
                margin: 5px;
            }
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 12px 20px;
                margin: 5px;
                border-radius: 10px;
                min-width: 120px;
            }
            QProgressBar {
                height: 20px;
                border: 2px solid #94ffed;
                border-radius: 10px;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #94ffed;
                border-radius: 8px;
            }
            QRadioButton {
                font-size: 18px;
                margin: 8px;
                padding: 8px;
                spacing: 8px;
                background-color: transparent;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
            QRadioButton:hover {
                color: #00695c;
                font-weight: bold;
            }
            QSpinBox {
                font-size: 18px;
                padding: 8px;
                min-height: 30px;
                border: 2px solid #94ffed;
                border-radius: 8px;
                background-color: white;
            }
            QComboBox {
                font-size: 16px;
                padding: 10px;
                min-height: 35px;
                border: 2px solid #94ffed;
                border-radius: 8px;
                background-color: white;
            }
            QComboBox:hover {
                border: 2px solid #00695c;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left: 2px solid #94ffed;
                border-radius: 0px;
            }
            QComboBox QAbstractItemView {
                font-size: 16px;
                background-color: white;
                selection-background-color: #94ffed;
                selection-color: #00695c;
                border: 2px solid #94ffed;
                outline: none;
            }
            QCheckBox {
                font-size: 18px;
                margin: 8px;
                padding: 8px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        self.title_label = QLabel("CLINICAL ASSESSMENT")
        self.title_label.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #00695c;
            padding: 10px;
        """)
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)
        
        # Step indicator
        self.step_label = QLabel("Step 1 of 7")
        self.step_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00695c;")
        self.step_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.step_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.total_steps)
        self.progress_bar.setValue(1)
        main_layout.addWidget(self.progress_bar)
        
        # Swipeable question container
        self.question_container = SwipeableWidget()
        self.question_container.setMinimumHeight(350)
        self.question_container.swipe_left.connect(self.next_step)
        self.question_container.swipe_right.connect(self.previous_step)
        
        self.question_layout = QVBoxLayout(self.question_container)
        self.question_layout.setSpacing(15)
        self.question_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.addWidget(self.question_container)
        
        # Visual indicator for swipe (small hint)
        swipe_hint = QLabel("← Swipe to navigate →")
        swipe_hint.setStyleSheet("font-size: 12px; color: #666; font-style: italic;")
        swipe_hint.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(swipe_hint)
        
        # Navigation buttons
        nav_widget = QtWidgets.QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setSpacing(20)
        
        self.back_button = QPushButton("← BACK")
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #ffd794; 
                color: #654700;
                padding: 15px 30px;
                font-size: 18px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #ffe7a8;
            }
            QPushButton:pressed {
                background-color: #dfc080;
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
                padding: 15px 30px;
                font-size: 18px;
                border-radius: 12px;
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
        
        nav_layout.addStretch(1)
        
        self.cancel_button = QPushButton("CANCEL")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9494; 
                color: #690000;
                padding: 15px 30px;
                font-size: 18px;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #ffa8a8;
            }
            QPushButton:pressed {
                background-color: #df8080;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_assessment)
        nav_layout.addWidget(self.cancel_button)
        
        main_layout.addWidget(nav_widget)
        self.setLayout(main_layout)
    
    def clear_question_area(self):
        """Completely clear all widgets from question area"""
        # Clear the layout properly
        while self.question_layout.count():
            item = self.question_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear the tracked widgets list
        self.current_widgets.clear()
        
        # Force immediate UI update
        QApplication.processEvents()
    
    def show_step(self, step):
        """Show a specific step in the wizard"""
        # Save current step answers before clearing
        self.save_answers()
        
        # Clear everything first - this ensures no text overlap
        self.clear_question_area()
        
        # Update step
        self.current_step = step
        self.progress_bar.setValue(step + 1)
        self.step_label.setText(f"Step {step + 1} of {self.total_steps}")
        
        # Update navigation buttons
        self.back_button.setVisible(step > 0)
        
        if step == self.total_steps - 1:
            self.next_button.setText("CALCULATE RESULTS")
        else:
            self.next_button.setText("NEXT →")
        
        # Show appropriate step content
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
            self.show_patient_info_step()
        elif step == 6:
            self.show_summary_step()
        
        # Force UI update
        QApplication.processEvents()
    
    def show_asymmetry_step(self):
        """Step 1: Asymmetry"""
        self.title_label.setText("A - ASYMMETRY")
        
        # Add stretch at top to center content vertically
        self.question_layout.addStretch(1)
        
        question = QLabel("Is the lesion asymmetrical?")
        question.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(question)
        self.current_widgets.append(question)
        
        description = QLabel("Asymmetry means if you draw a line through the middle,\nthe two halves don't match.")
        description.setStyleSheet("font-size: 16px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(description)
        self.current_widgets.append(description)
        
        self.question_layout.addSpacing(20)
        
        # Options with larger touch area
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(10)
        
        self.asymmetry_yes = QRadioButton("YES - Asymmetrical")
        self.asymmetry_no = QRadioButton("NO - Symmetrical")
        self.asymmetry_no.setChecked(True)
        
        for rb in [self.asymmetry_yes, self.asymmetry_no]:
            rb.setStyleSheet("""
                QRadioButton {
                    font-size: 20px;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #94ffed;
                    border-radius: 8px;
                    margin: 2px;
                }
                QRadioButton:hover {
                    background-color: #e8fff8;
                    border: 2px solid #00695c;
                }
                QRadioButton:checked {
                    background-color: #94ffed;
                    border: 2px solid #00695c;
                    font-weight: bold;
                }
            """)
        
        option_layout.addWidget(self.asymmetry_yes)
        option_layout.addWidget(self.asymmetry_no)
        option_layout.addStretch(1)
        
        self.question_layout.addWidget(option_group)
        self.current_widgets.extend([option_group, self.asymmetry_yes, self.asymmetry_no])
        
        self.question_layout.addStretch(2)
    
    def show_border_step(self):
        """Step 2: Border"""
        self.title_label.setText("B - BORDER")
        
        self.question_layout.addStretch(1)
        
        question = QLabel("Is the border irregular, ragged, or notched?")
        question.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(question)
        self.current_widgets.append(question)
        
        description = QLabel("Irregular borders look like a coastline with bays,\nnot a smooth, round circle.")
        description.setStyleSheet("font-size: 16px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(description)
        self.current_widgets.append(description)
        
        self.question_layout.addSpacing(20)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(10)
        
        self.border_yes = QRadioButton("YES - Irregular border")
        self.border_no = QRadioButton("NO - Smooth border")
        self.border_no.setChecked(True)
        
        for rb in [self.border_yes, self.border_no]:
            rb.setStyleSheet("""
                QRadioButton {
                    font-size: 20px;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #94ffed;
                    border-radius: 8px;
                    margin: 2px;
                }
                QRadioButton:hover {
                    background-color: #e8fff8;
                    border: 2px solid #00695c;
                }
                QRadioButton:checked {
                    background-color: #94ffed;
                    border: 2px solid #00695c;
                    font-weight: bold;
                }
            """)
        
        option_layout.addWidget(self.border_yes)
        option_layout.addWidget(self.border_no)
        option_layout.addStretch(1)
        
        self.question_layout.addWidget(option_group)
        self.current_widgets.extend([option_group, self.border_yes, self.border_no])
        
        self.question_layout.addStretch(2)
    
    def show_color_step(self):
        """Step 3: Color"""
        self.title_label.setText("C - COLOR")
        
        self.question_layout.addStretch(1)
        
        question = QLabel("How many colors does the lesion have?")
        question.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(question)
        self.current_widgets.append(question)
        
        description = QLabel("More colors = higher risk. Look for shades of brown, black, red, white, or blue.")
        description.setStyleSheet("font-size: 16px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(description)
        self.current_widgets.append(description)
        
        self.question_layout.addSpacing(20)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(10)
        
        self.color_single = QRadioButton("Single color (brown, tan, or black)")
        self.color_two = QRadioButton("2-3 colors")
        self.color_many = QRadioButton("Many different colors")
        self.color_single.setChecked(True)
        
        for rb in [self.color_single, self.color_two, self.color_many]:
            rb.setStyleSheet("""
                QRadioButton {
                    font-size: 20px;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #94ffed;
                    border-radius: 8px;
                    margin: 2px;
                }
                QRadioButton:hover {
                    background-color: #e8fff8;
                    border: 2px solid #00695c;
                }
                QRadioButton:checked {
                    background-color: #94ffed;
                    border: 2px solid #00695c;
                    font-weight: bold;
                }
            """)
        
        option_layout.addWidget(self.color_single)
        option_layout.addWidget(self.color_two)
        option_layout.addWidget(self.color_many)
        option_layout.addStretch(1)
        
        self.question_layout.addWidget(option_group)
        self.current_widgets.extend([option_group, self.color_single, self.color_two, self.color_many])
        
        self.question_layout.addStretch(2)
    
    def show_diameter_step(self):
        """Step 4: Diameter"""
        self.title_label.setText("D - DIAMETER")
        
        self.question_layout.addStretch(1)
        
        question = QLabel("What is the size of the lesion?")
        question.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(question)
        self.current_widgets.append(question)
        
        description = QLabel("Measure the widest part. A pencil eraser is about 6mm.")
        description.setStyleSheet("font-size: 16px; font-style: italic; color: #666; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(description)
        self.current_widgets.append(description)
        
        self.question_layout.addSpacing(20)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(10)
        
        self.diameter_small = QRadioButton("Small (less than 6mm)")
        self.diameter_medium = QRadioButton("Medium (6-10mm)")
        self.diameter_large = QRadioButton("Large (more than 10mm)")
        self.diameter_small.setChecked(True)
        
        for rb in [self.diameter_small, self.diameter_medium, self.diameter_large]:
            rb.setStyleSheet("""
                QRadioButton {
                    font-size: 20px;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #94ffed;
                    border-radius: 8px;
                    margin: 2px;
                }
                QRadioButton:hover {
                    background-color: #e8fff8;
                    border: 2px solid #00695c;
                }
                QRadioButton:checked {
                    background-color: #94ffed;
                    border: 2px solid #00695c;
                    font-weight: bold;
                }
            """)
        
        option_layout.addWidget(self.diameter_small)
        option_layout.addWidget(self.diameter_medium)
        option_layout.addWidget(self.diameter_large)
        option_layout.addStretch(1)
        
        self.question_layout.addWidget(option_group)
        self.current_widgets.extend([option_group, self.diameter_small, self.diameter_medium, self.diameter_large])
        
        self.question_layout.addStretch(2)
    
    def show_evolution_step(self):
        """Step 5: Evolution"""
        self.title_label.setText("E - EVOLUTION")
        
        self.question_layout.addStretch(1)
        
        question = QLabel("Has the lesion changed over time?")
        question.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(question)
        self.current_widgets.append(question)
        
        description = QLabel("Recent changes in size, shape, or color are a major warning sign!")
        description.setStyleSheet("font-size: 16px; font-style: italic; color: #d32f2f; font-weight: bold; margin: 5px;")
        description.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(description)
        self.current_widgets.append(description)
        
        self.question_layout.addSpacing(20)
        
        option_group = QtWidgets.QGroupBox("Select one:")
        option_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        option_layout = QVBoxLayout(option_group)
        option_layout.setSpacing(10)
        
        self.evolution_no = QRadioButton("NO CHANGE - Stable for years")
        self.evolution_slow = QRadioButton("SLOW CHANGE - Over months/years")
        self.evolution_fast = QRadioButton("RAPID CHANGE - Weeks/months")
        self.evolution_no.setChecked(True)
        
        for rb in [self.evolution_no, self.evolution_slow, self.evolution_fast]:
            rb.setStyleSheet("""
                QRadioButton {
                    font-size: 20px;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #94ffed;
                    border-radius: 8px;
                    margin: 2px;
                }
                QRadioButton:hover {
                    background-color: #e8fff8;
                    border: 2px solid #00695c;
                }
                QRadioButton:checked {
                    background-color: #94ffed;
                    border: 2px solid #00695c;
                    font-weight: bold;
                }
            """)
        
        option_layout.addWidget(self.evolution_no)
        option_layout.addWidget(self.evolution_slow)
        option_layout.addWidget(self.evolution_fast)
        option_layout.addStretch(1)
        
        self.question_layout.addWidget(option_group)
        self.current_widgets.extend([option_group, self.evolution_no, self.evolution_slow, self.evolution_fast])
        
        self.question_layout.addStretch(2)
    
    def show_patient_info_step(self):
        """Step 6: Patient Information"""
        self.title_label.setText("PATIENT INFORMATION")
        
        self.question_layout.addStretch(1)
        
        question = QLabel("Tell us about the patient")
        question.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        question.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(question)
        self.current_widgets.append(question)
        
        self.question_layout.addSpacing(10)
        
        # Create a scroll area for patient info to ensure all content is accessible
        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setStyleSheet("""
            QScrollArea {
                border: 2px solid #94ffed;
                border-radius: 10px;
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)
        
        info_widget = QtWidgets.QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setSpacing(15)
        info_layout.setContentsMargins(15, 15, 15, 15)
        
        # Age with larger touch area
        age_group = QtWidgets.QGroupBox("Age")
        age_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
        """)
        age_layout = QHBoxLayout(age_group)
        age_layout.setSpacing(20)
        
        age_label = QLabel("Patient age:")
        age_label.setStyleSheet("font-size: 18px; padding: 5px;")
        
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setRange(1, 120)
        self.age_spinbox.setValue(40)
        self.age_spinbox.setStyleSheet("""
            QSpinBox {
                font-size: 20px;
                padding: 10px;
                min-height: 40px;
                border: 2px solid #94ffed;
                border-radius: 8px;
                background-color: white;
            }
            QSpinBox:hover {
                border: 2px solid #00695c;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 30px;
                height: 20px;
            }
        """)
        
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_spinbox)
        age_layout.addStretch(1)
        info_layout.addWidget(age_group)
        
        # Skin Type with properly formatted dropdown
        skin_group = QtWidgets.QGroupBox("Skin Type (Fitzpatrick Scale)")
        skin_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
        """)
        skin_layout = QVBoxLayout(skin_group)
        
        skin_description = QLabel("Select the skin type that best matches:")
        skin_description.setStyleSheet("font-size: 16px; color: #666; margin: 5px;")
        skin_description.setWordWrap(True)
        skin_layout.addWidget(skin_description)
        
        self.skin_combo = QComboBox()
        self.skin_combo.addItems([
            "Type I - Always burns, never tans (very fair skin, red/blonde hair)",
            "Type II - Usually burns, tans minimally (fair skin, light eyes)",
            "Type III - Sometimes burns, tans gradually (fair to beige skin)",
            "Type IV - Rarely burns, tans well (olive or light brown skin)",
            "Type V - Very rarely burns, tans easily (brown skin)",
            "Type VI - Never burns (deeply pigmented dark brown to black skin)"
        ])
        self.skin_combo.setCurrentIndex(2)
        self.skin_combo.setStyleSheet("""
            QComboBox {
                font-size: 16px;
                padding: 12px;
                min-height: 45px;
                border: 2px solid #94ffed;
                border-radius: 8px;
                background-color: white;
            }
            QComboBox:hover {
                border: 2px solid #00695c;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 40px;
                border-left: 2px solid #94ffed;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }
            QComboBox::down-arrow {
                width: 20px;
                height: 20px;
            }
            QComboBox QAbstractItemView {
                font-size: 16px;
                background-color: white;
                selection-background-color: #94ffed;
                selection-color: #00695c;
                border: 2px solid #94ffed;
                outline: none;
                min-height: 40px;
            }
        """)
        skin_layout.addWidget(self.skin_combo)
        info_layout.addWidget(skin_group)
        
        # Risk factors with larger checkboxes
        risk_group = QtWidgets.QGroupBox("Risk Factors")
        risk_group.setStyleSheet("""
            QGroupBox {
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #94ffed;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
        """)
        risk_layout = QVBoxLayout(risk_group)
        
        self.family_check = QCheckBox("Family history of skin cancer")
        self.sunburn_check = QCheckBox("History of severe sunburns")
        
        for cb in [self.family_check, self.sunburn_check]:
            cb.setStyleSheet("""
                QCheckBox {
                    font-size: 18px;
                    padding: 12px;
                    background-color: white;
                    border: 1px solid #94ffed;
                    border-radius: 8px;
                    margin: 2px;
                }
                QCheckBox:hover {
                    background-color: #e8fff8;
                    border: 2px solid #00695c;
                }
                QCheckBox:checked {
                    background-color: #94ffed;
                    border: 2px solid #00695c;
                }
                QCheckBox::indicator {
                    width: 24px;
                    height: 24px;
                }
            """)
        
        risk_layout.addWidget(self.family_check)
        risk_layout.addWidget(self.sunburn_check)
        risk_layout.addStretch(1)
        info_layout.addWidget(risk_group)
        
        info_scroll.setWidget(info_widget)
        self.question_layout.addWidget(info_scroll)
        self.current_widgets.extend([info_scroll, info_widget, age_group, skin_group, risk_group,
                                     self.age_spinbox, self.skin_combo, self.family_check, self.sunburn_check])
        
        self.question_layout.addStretch(1)
    
    def show_summary_step(self):
        """Step 7: Summary"""
        self.title_label.setText("SUMMARY")
        
        # Save answers
        self.save_answers()
        
        self.question_layout.addStretch(1)
        
        summary_text = QLabel("Review your answers:")
        summary_text.setStyleSheet("font-size: 24px; font-weight: bold; color: #00695c; margin: 10px;")
        summary_text.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(summary_text)
        self.current_widgets.append(summary_text)
        
        self.question_layout.addSpacing(10)
        
        # Display summary in a scrollable text area
        summary_display = QTextEdit()
        summary_display.setReadOnly(True)
        summary_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 2px solid #94ffed;
                border-radius: 10px;
                padding: 15px;
                font-size: 16px;
                min-height: 250px;
            }
        """)
        
        # Build summary
        summary_html = "<h3 style='color: #00695c;'>ABCDE Assessment:</h3>"
        summary_html += f"<p><b>A. Asymmetry:</b> {'YES' if self.abcde_answers['asymmetry'] else 'NO'}</p>"
        summary_html += f"<p><b>B. Border:</b> {'YES' if self.abcde_answers['border'] else 'NO'}</p>"
        summary_html += f"<p><b>C. Color:</b> {self.abcde_answers['color'].title()}</p>"
        summary_html += f"<p><b>D. Diameter:</b> {self.abcde_answers['diameter'].title()}</p>"
        summary_html += f"<p><b>E. Evolution:</b> {self.abcde_answers['evolution'].title().replace('_', ' ')}</p>"
        
        summary_html += "<h3 style='color: #00695c; margin-top: 20px;'>Patient Information:</h3>"
        summary_html += f"<p><b>Age:</b> {self.patient_data['age']}</p>"
        
        # Get the full skin type text
        skin_types = [
            "Type I - Always burns, never tans (very fair skin, red/blonde hair)",
            "Type II - Usually burns, tans minimally (fair skin, light eyes)",
            "Type III - Sometimes burns, tans gradually (fair to beige skin)",
            "Type IV - Rarely burns, tans well (olive or light brown skin)",
            "Type V - Very rarely burns, tans easily (brown skin)",
            "Type VI - Never burns (deeply pigmented dark brown to black skin)"
        ]
        summary_html += f"<p><b>Skin Type:</b> {skin_types[self.patient_data['skin_type']]}</p>"
        summary_html += f"<p><b>Family History:</b> {'YES' if self.patient_data['family_history'] else 'NO'}</p>"
        summary_html += f"<p><b>Sunburn History:</b> {'YES' if self.patient_data['sunburn_history'] else 'NO'}</p>"
        
        summary_display.setHtml(summary_html)
        self.question_layout.addWidget(summary_display)
        self.current_widgets.append(summary_display)
        
        note = QLabel("Click 'CALCULATE RESULTS' to generate your comprehensive risk assessment.")
        note.setStyleSheet("font-size: 16px; font-style: italic; color: #666; margin-top: 10px;")
        note.setAlignment(Qt.AlignCenter)
        self.question_layout.addWidget(note)
        self.current_widgets.append(note)
        
        self.question_layout.addStretch(2)
    
    def save_answers(self):
        """Save answers from current step"""
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
        elif self.current_step == 5 and hasattr(self, 'age_spinbox'):
            self.patient_data['age'] = self.age_spinbox.value()
            self.patient_data['skin_type'] = self.skin_combo.currentIndex()
            self.patient_data['family_history'] = self.family_check.isChecked()
            self.patient_data['sunburn_history'] = self.sunburn_check.isChecked()
    
    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.show_step(self.current_step - 1)
    
    def next_step(self):
        """Go to next step or calculate results"""
        # Save current answers
        self.save_answers()
        
        if self.current_step < self.total_steps - 1:
            self.show_step(self.current_step + 1)
        else:
            # Calculate results
            self.calculate_results()
    
    def calculate_results(self):
        """Calculate final risk score"""
        # Stop blinking before showing results
        if self.parent_app:
            self.parent_app.stop_yellow_blinking()
        
        # Calculate ABCDE score
        abcde_score = 0
        
        # A - Asymmetry
        if self.abcde_answers['asymmetry']:
            abcde_score += 1
        
        # B - Border
        if self.abcde_answers['border']:
            abcde_score += 1
        
        # C - Color
        if self.abcde_answers['color'] == 'two':
            abcde_score += 1
        elif self.abcde_answers['color'] == 'many':
            abcde_score += 2
        
        # D - Diameter
        if self.abcde_answers['diameter'] in ['medium', 'large']:
            abcde_score += 1
        
        # E - Evolution
        if self.abcde_answers['evolution'] == 'slow':
            abcde_score += 1
        elif self.abcde_answers['evolution'] == 'fast':
            abcde_score += 2
        
        # Patient Risk Score
        patient_score = 0
        if self.patient_data['age'] > 50:
            patient_score += 1
        if self.patient_data['skin_type'] < 2:
            patient_score += 1
        if self.patient_data['family_history']:
            patient_score += 1
        if self.patient_data['sunburn_history']:
            patient_score += 1
        
        # CNN Risk Weight
        if "melanoma" in self.cnn_prediction.lower() or "carcinoma" in self.cnn_prediction.lower():
            cnn_weight = 3
        elif self.cnn_prediction.lower() == "normal":
            cnn_weight = 0
        else:
            cnn_weight = 1
        
        # Combined Risk
        total_risk = (
            (abcde_score / 8.0) * 40 +
            (patient_score / 4.0) * 20 +
            (cnn_weight * self.cnn_confidence) * 40
        )
        
        # Store results
        self.final_results = {
            'abcde_score': abcde_score,
            'patient_score': patient_score,
            'total_risk': total_risk,
            'cnn_prediction': self.cnn_prediction,
            'cnn_confidence': self.cnn_confidence
        }
        
        # Determine LED color
        if total_risk >= 70:
            self.final_results['led_color'] = "RED"
            self.final_results['recommendation'] = """• URGENT: Dermatology referral (within 1-2 weeks)
• Do not delay evaluation
• Monitor for any changes
• Avoid sun exposure"""
        elif total_risk >= 40:
            self.final_results['led_color'] = "YELLOW"
            self.final_results['recommendation'] = """• Schedule dermatology appointment (4-6 weeks)
• Monitor monthly for changes
• Practice sun protection
• Use SPF 30+ daily"""
        else:
            self.final_results['led_color'] = "GREEN"
            self.final_results['recommendation'] = """• Continue regular self-checks
• Annual skin examination recommended
• Practice sun safety
• Use SPF 15+ daily"""
        
        # Show success pattern
        if self.parent_app:
            self.parent_app.show_green_completion_pattern()
        
        # Close dialog
        self.accept()
    
    def cancel_assessment(self):
        """Cancel assessment"""
        reply = QMessageBox.question(
            self, 'Cancel', 'Cancel assessment? All answers will be lost.',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            if self.parent_app:
                self.parent_app.stop_yellow_blinking()
            self.reject()
    
    def reject(self):
        """Handle window close (X button)"""
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
            
            # CRITICAL: Explicitly request RGB888 format for correct colors
            try:
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    controls={"FrameRate": 30}
                )
                self.picam2.configure(config)
                print("Camera configured with RGB888 format")
            except:
                # Fallback to simple config but still try to get RGB
                print("Falling back to simple configuration")
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480)}
                )
                self.picam2.configure(config)
            
            self.picam2.start()
            print("Camera started successfully")
            
            while self.running:
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None:
                        # Ensure RGB format
                        if len(frame.shape) == 2:  # Grayscale
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 4:  # RGBA
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        elif frame.shape[2] == 3:
                            # Check if the frame might be in BGR format
                            # Sample the center pixel to detect color order
                            h, w, _ = frame.shape
                            sample_y, sample_x = h // 2, w // 2
                            sample = frame[sample_y, sample_x]
                            
                            # If blue channel is significantly higher than red, probably BGR
                            # Typical skin has red > blue, so if blue > red + 30, it's likely BGR
                            if sample[2] > sample[0] + 30:  # Blue > Red by threshold
                                print("Detected BGR format, converting to RGB")
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
            # Provide a dummy black frame
            import numpy as np
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
        
        # Turn all LEDs off initially
        turn_off_leds()
        
        # Blinking control
        self.blink_timer = None
        self.blink_state = False
        self.blink_count = 0
        self.max_blinks = 20  # 10 seconds at 500ms interval (on/off = 1 second per blink cycle)
        
        # Classes
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
    
    def initUI(self):
        self.setWindowTitle("NOMA AI - Skin Analysis")
        self.showFullScreen()
        
        # Get screen dimensions for touch calibration
        screen_geometry = QApplication.primaryScreen().geometry()
        self.screen_height = screen_geometry.height()
        print(f"Screen height: {self.screen_height}")
        
        # Set the green background for the entire app
        self.setStyleSheet("""
            QMainWindow {
                background-color: #b8fcbf;
            }
            QWidget {
                background-color: #b8fcbf;
            }
            QScrollArea {
                border: none;
                background-color: #b8fcbf;
            }
        """)
        
        # Create scroll area for comprehensive results
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Create a central widget with proper touch-friendly layout
        central_widget = QtWidgets.QWidget()
        self.scroll_area.setWidget(central_widget)
        self.setCentralWidget(self.scroll_area)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # NOMA AI Title
        title_label = QLabel("NOMA AI - Advanced Skin Analysis")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
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

        # LARGER Camera preview - centered with container
        self.image_label = QLabel("Loading camera feed...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)  # Larger preview
        self.image_label.setMaximumSize(400, 300)  # Larger preview
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 3px solid #94ffed;
                border-radius: 10px;
                padding: 5px;
                margin: 5px;
            }
        """)
        
        # Center the camera preview in its own container
        camera_container = QtWidgets.QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.image_label)
        layout.addWidget(camera_container)

        # HUGE Classify button with improved touch area
        self.classify_button = QPushButton("CAPTURE AND ANALYZE COMPREHENSIVELY")
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
            QPushButton#classify_button:hover {
                background-color: #a8fff0;
                border: 4px solid #94ffed;
            }
            QPushButton#classify_button:pressed {
                background-color: #80dfd0;
                color: #004d40;
            }
            QPushButton#classify_button:disabled {
                background-color: #c8fcf5;
                color: #80a09c;
            }
        """)
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)
        layout.addWidget(self.classify_button)

        # LED GUIDE Button
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
            QPushButton:hover {
                background-color: #ffe7a8;
                border: 4px solid #ffd794;
            }
            QPushButton:pressed {
                background-color: #dfc080;
                color: #402d00;
            }
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

        # Results display area
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

        # Analysis image display
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
        layout.addWidget(self.analysis_label)

        # Large Shutdown button
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
            QPushButton#shutdown_button:hover {
                background-color: #ffa8a8;
                border: 4px solid #ff9494;
            }
            QPushButton#shutdown_button:pressed {
                background-color: #df8080;
                color: #400000;
            }
        """)
        self.shutdown_button.clicked.connect(self.shutdown_device)
        layout.addWidget(self.shutdown_button)

        # Safety disclaimer
        disclaimer_label = QLabel("*Not a medical diagnosis. For educational use only.*")
        disclaimer_label.setAlignment(Qt.AlignCenter)
        disclaimer_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-style: italic;
                color: #666;
                padding: 5px;
                margin: 5px;
            }
        """)
        layout.addWidget(disclaimer_label)

        # Add stretch at bottom to help centering
        layout.addStretch(1)
    
    def show_led_guide(self):
        """Show LED status guide in a scrollable dialog with exit button"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("LED Status Guide")
        dialog.setMinimumSize(500, 400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #b8fcbf;
            }
            QTextEdit {
                background-color: white;
                border: 2px solid #94ffed;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                line-height: 1.4;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        
        # Title
        title = QLabel("LED STATUS GUIDE")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00695c; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Scrollable text area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml("""
        <h3>YELLOW BLINKING:</h3>
        <ul>
        <li>System is analyzing your image</li>
        <li>Please wait while processing completes</li>
        <li>This takes about 5-10 seconds</li>
        </ul>
        
        <h3>SOLID RED:</h3>
        <ul>
        <li>MALIGNANT detection</li>
        <li>High risk potential</li>
        <li>Consult dermatologist promptly</li>
        <li>Examples: Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma</li>
        </ul>
        
        <h3>SOLID YELLOW:</h3>
        <ul>
        <li>BENIGN detection</li>
        <li>Moderate risk or uncertain</li>
        <li>Monitor regularly</li>
        <li>Examples: Moles, Eczema, Psoriasis, Acne, Rosacea</li>
        </ul>
        
        <h3>SOLID GREEN:</h3>
        <ul>
        <li>NORMAL skin</li>
        <li>Low risk</li>
        <li>Continue regular self-checks</li>
        <li>Example: Healthy skin tissue</li>
        </ul>
        
        <h3>ADDITIONAL NOTES:</h3>
        <ul>
        <li>All LEDs turn off automatically after 8 seconds</li>
        <li>Low confidence results will show yellow LED with recommendation to retake image</li>
        <li>Ensure good lighting and clear focus for best results</li>
        <li>Position lesion clearly in the center of the camera view</li>
        </ul>
        """)
        layout.addWidget(text_edit)
        
        # Exit button
        exit_button = QPushButton("CLOSE GUIDE")
        exit_button.setMinimumHeight(50)
        exit_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
                background-color: #94ffed;
                color: #00695c;
                border: 3px solid #80dfd0;
                border-radius: 15px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #a8fff0;
                border: 3px solid #94ffed;
            }
            QPushButton:pressed {
                background-color: #80dfd0;
                color: #004d40;
            }
        """)
        exit_button.clicked.connect(dialog.accept)
        layout.addWidget(exit_button)
        
        dialog.exec_()
    
    # LED Methods
    def start_yellow_blinking_for_dialog(self):
        """Start yellow LED blinking using a timer (max 10 seconds)"""
        self.stop_yellow_blinking()
        self.blink_state = False
        self.blink_count = 0
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._blink_yellow)
        self.blink_timer.start(500)  # 500 ms interval
    
    def _blink_yellow(self):
        """Toggle yellow LED with max 10 second limit"""
        self.blink_count += 1
        if self.blink_count > self.max_blinks:  # Stop after max_blinks (20 = 10 seconds)
            self.stop_yellow_blinking()
            return
        
        self.blink_state = not self.blink_state
        set_leds(yellow=self.blink_state)
    
    def stop_yellow_blinking(self):
        """Stop blinking and turn off yellow LED"""
        if self.blink_timer:
            self.blink_timer.stop()
            self.blink_timer = None
        set_leds(yellow=False)
        self.blink_count = 0
    
    def show_green_completion_pattern(self):
        """Show green completion pattern"""
        # Ensure blinking is stopped
        self.stop_yellow_blinking()
        # Pattern: two short blinks
        set_leds(green=True)
        QTimer.singleShot(300, lambda: set_leds(green=False))
        QTimer.singleShot(600, lambda: set_leds(green=True))
        QTimer.singleShot(900, lambda: set_leds(green=False))
        QTimer.singleShot(1200, lambda: set_leds(green=True))
        QTimer.singleShot(1500, lambda: set_leds(green=False))
    
    # Grad-CAM
    def generate_grad_cam_heatmap(self, image, predicted_class, confidence):
        """Generate Grad-CAM heatmap"""
        try:
            img_array = np.array(image.resize((224, 224)))
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Simple heatmap
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img_array, 0.5, heatmap, 0.5, 0)
            
            return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return image
    
    # Classification
    def classify_image(self):
        if self.is_classifying:
            return
        
        self.is_classifying = True
        self.classify_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.results_label.setText("Starting analysis...")
        QApplication.processEvents()
        
        try:
            # Capture frame
            frame = self.camera_thread.get_latest_frame()
            if frame is None:
                QMessageBox.warning(self, "Warning", "No camera feed")
                return
            
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            image = Image.fromarray(frame)
            img_array = self.preprocess_image(image)
            
            # AI inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_class = self.classes[class_index]
            
            if confidence < 0.6:
                QMessageBox.warning(self, "Low Confidence", 
                    f"Confidence too low ({confidence:.1%}).\nPlease retake with better lighting.")
                self.set_leds_timed(False, True, False)
                self.results_label.setText(f"Low confidence ({confidence:.1%}). Please retake image.")
                return
            
            # Launch step-by-step assessment
            dialog = StepByStepClinicalAssessor(self, predicted_class, confidence)
            
            if dialog.exec_():
                # Assessment completed
                results = dialog.final_results
                
                # Generate Grad-CAM
                heatmap = self.generate_grad_cam_heatmap(image, predicted_class, confidence)
                heatmap_array = np.array(heatmap)
                heatmap_qimage = QtGui.QImage(
                    heatmap_array.data,
                    heatmap_array.shape[1],
                    heatmap_array.shape[0],
                    heatmap_array.strides[0],
                    QtGui.QImage.Format_RGB888
                )
                pixmap = QtGui.QPixmap.fromImage(heatmap_qimage).scaled(
                    400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.analysis_label.setPixmap(pixmap)
                
                # Set LED
                led_color = results['led_color']
                if led_color == "RED":
                    self.set_leds_timed(True, False, False)
                elif led_color == "YELLOW":
                    self.set_leds_timed(False, True, False)
                else:
                    self.set_leds_timed(False, False, True)
                
                # Show results
                result_text = f"""
                🏥 COMPREHENSIVE ANALYSIS COMPLETE
                
                AI DETECTION:
                • Condition: {results['cnn_prediction']}
                • Confidence: {results['cnn_confidence']:.1%}
                
                CLINICAL ASSESSMENT:
                • ABCDE Score: {results['abcde_score']}/8
                • Patient Risk: {results['patient_score']}/4
                • Total Risk: {results['total_risk']:.0f}/100
                
                RISK LEVEL: {led_color}
                
                RECOMMENDATION:
                {results['recommendation']}
                
                *This is a screening tool only.
                Always consult a healthcare professional.*
                """
                self.results_label.setText(result_text)
            else:
                # User cancelled
                self.results_label.setText("Assessment cancelled.")
                turn_off_leds()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.results_label.setText(f"Error: {str(e)}")
        
        finally:
            self.is_classifying = False
            self.classify_button.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    # Camera
    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_camera_feed)
        self.camera_thread.start()
    
    def update_camera_feed(self, qt_image):
        """Update camera feed with correct colors"""
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
            400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)
        # Force immediate UI update
        self.image_label.repaint()
    
    # Model
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
        """Adapt preprocessing to model's expected dtype and shape."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image.resize((224, 224)), dtype=np.float32)
        
        # Get expected dtype from model
        expected_dtype = self.input_details[0]['dtype']
        expected_shape = self.input_details[0]['shape']
        
        # Normalize to [-1,1] if model expects float32 (typical for MobileNetV3 without rescaling)
        if expected_dtype == np.float32:
            img_array = (img_array / 127.5) - 1.0
        # If model expects uint8, leave as [0,255] (the model should have internal rescaling)
        elif expected_dtype == np.uint8:
            img_array = img_array.astype(np.uint8)
        else:
            # Fallback: assume float32 normalization
            img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension if needed
        if len(expected_shape) == 4 and len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        elif len(expected_shape) == 3 and len(img_array.shape) == 4:
            img_array = img_array.squeeze(0)
        
        # Ensure correct dtype
        img_array = img_array.astype(expected_dtype)
        return img_array
    
    # LED Control
    def set_leds(self, red=False, yellow=False, green=False):
        set_leds(red=red, yellow=yellow, green=green)
    
    def set_leds_timed(self, red=False, yellow=False, green=False):
        set_leds(red=red, yellow=yellow, green=green)
        QTimer.singleShot(8000, turn_off_leds)
    
    def turn_off_leds(self):
        turn_off_leds()
    
    # GPIO Reset
    def reset_gpio(self):
        """Reset GPIO: turn off LEDs without releasing pins"""
        reply = QMessageBox.question(
            self, "Reset GPIO", 
            "Turn off all LEDs?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.turn_off_leds()
            QMessageBox.information(self, "GPIO Reset", "All LEDs turned off.")
    
    # Shutdown
    def shutdown_device(self):
        reply = QMessageBox.question(
            self, "Shutdown", "Shut down the device?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Cleanup GPIO before shutdown
            led_controller.cleanup()
            os.system("sudo shutdown now")
    
    def closeEvent(self, event):
        """Handle application close"""
        print("Closing application...")
        self.stop_yellow_blinking()
        self.camera_thread.stop()
        led_controller.cleanup()
        event.accept()

# Main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    
    # Force fullscreen on Raspberry Pi
    app.setOverrideCursor(Qt.BlankCursor)
    
    ex = NomaAIApp()
    ex.show()
    
    # Ensure cleanup on exit
    app.aboutToQuit.connect(led_controller.cleanup)
    
    sys.exit(app.exec_())
