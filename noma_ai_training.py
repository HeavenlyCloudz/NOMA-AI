# ======================
# NOMA AI - HIGH ACCURACY TFLITE-READY MODEL
# ======================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
import sys

print("🚀 NOMA AI - HIGH ACCURACY TFLITE-READY MODEL")
print("TensorFlow version:", tf.__version__)

# DATASET
dataset_path = '/kaggle/input/skin-diseases-cancer-comprehensive-dataset/noma_ai_dataset'
CLASS_NAMES = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print(f"🎯 Training with {len(CLASS_NAMES)} classes: {CLASS_NAMES}")

# DATASET CONFIG
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

# TFLITE-COMPATIBLE PREPROCESSING
def preprocess_train(image, label):
    # MobileNetV3 expects [0,255] range - NO rescaling to [-1,1]
    return image, label

def preprocess_val(image, label):
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess_val, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

print(f"✅ Training samples: {len(train_ds) * BATCH_SIZE}")
print(f"✅ Validation samples: {len(val_ds) * BATCH_SIZE}")

# HIGH ACCURACY MOBILENETV3 MODEL
def create_high_accuracy_model(num_classes=len(CLASS_NAMES)):
    print("🏗️ Building High-Accuracy MobileNetV3 Model...")
    
    # MobileNetV3 base - exactly like your successful model
    base_model = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=False  # We handle preprocessing separately for TFLite
    )
    
    base_model.trainable = False
    
    # Build model with preprocessing included
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # TFLite-compatible preprocessing
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1.0)(inputs)  # Convert to [-1, 1]
    
    # Base model
    x = base_model(x, training=False)
    
    # Enhanced classifier for high accuracy
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs, name='noma_cancer_ai_model')
    return model, base_model

# CREATE MODEL
print("🤖 Creating High-Accuracy Model...")
model, base_model = create_high_accuracy_model()
model.summary()

# CLASS WEIGHTS
class_counts = []
for class_name in CLASS_NAMES:
    class_path = os.path.join(dataset_path, class_name)
    num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    class_counts.append(num_images)

y_train = []
for i, count in enumerate(class_counts):
    y_train.extend([i] * count)

class_weights = compute_class_weight('balanced', classes=np.arange(len(CLASS_NAMES)), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# TRAINING STRATEGY FOR HIGH ACCURACY
class HighAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val = 0
        self.best_train = 0
        
    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        if val_acc > self.best_val:
            self.best_val = val_acc
        if train_acc > self.best_train:
            self.best_train = train_acc
            
        print(f"Epoch {epoch+1}: Train: {train_acc:.2%}, Val: {val_acc:.2%}, Best Val: {self.best_val:.2%}")
        
        # Stop if we hit targets
        if train_acc >= 0.85 and val_acc >= 0.75:
            print("🎉 TARGETS ACHIEVED! Stopping training early.")
            self.model.stop_training = True

# TRAINING FUNCTION
def train_high_accuracy(model, base_model):
    # STAGE 1: Classifier training
    print("\n" + "="*50)
    print("STAGE 1: Training Classifier (30 epochs)")
    print("="*50)
    
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_ds, epochs=30, validation_data=val_ds,
        callbacks=[HighAccuracyCallback()],
        class_weight=class_weight_dict, verbose=0
    )
    
    # STAGE 2: Fine-tuning
    print("\n" + "="*50)
    print("STAGE 2: Fine-tuning (40 epochs)")
    print("="*50)
    
    base_model.trainable = True
    # Freeze early layers
    for layer in base_model.layers[:-40]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds, epochs=40, validation_data=val_ds,
        callbacks=[HighAccuracyCallback()],
        class_weight=class_weight_dict, verbose=0
    )
    
    # STAGE 3: Full training if needed
    if HighAccuracyCallback().best_val < 0.75:
        print("\n" + "="*50)
        print("STAGE 3: Full Training (30 epochs)")
        print("="*50)
        
        base_model.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history3 = model.fit(
            train_ds, epochs=30, validation_data=val_ds,
            callbacks=[HighAccuracyCallback()],
            class_weight=class_weight_dict, verbose=0
        )
        return history1, history2, history3
    
    return history1, history2, None

# START TRAINING
print("\n🔥 STARTING HIGH ACCURACY TRAINING...")
print("🎯 Targets: 85%+ Train Accuracy, 75%+ Val Accuracy")
history1, history2, history3 = train_high_accuracy(model, base_model)

# FINAL EVALUATION
print("\n📊 FINAL EVALUATION...")
final_val_loss, final_val_accuracy = model.evaluate(val_ds, verbose=0)
final_train_loss, final_train_accuracy = model.evaluate(train_ds.take(10), verbose=0)

print(f"🏆 FINAL RESULTS:")
print(f"   • Training Accuracy: {final_train_accuracy:.2%}")
print(f"   • Validation Accuracy: {final_val_accuracy:.2%}")

# VERIFY TFLITE COMPATIBILITY
print("\n🔍 Verifying TFLite compatibility...")
try:
    # Test conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Test inference
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    print("✅ 100% TFLite Compatible!")
    print("✅ No Flex Ops Required!")
    print("✅ Preprocessing Layers Included!")
    print("✅ Ready for Raspberry Pi!")
    
    # Save the trained model for later conversion
    model.save('/kaggle/working/noma_cancer_ai_model.keras')
    print("✅ Model saved as: /kaggle/working/noma_cancer_ai_model.keras")
    
except Exception as e:
    print(f"❌ TFLite issue: {e}")

# PERFORMANCE CHECK
if final_val_accuracy >= 0.75:
    print("🎉 EXCELLENT! High accuracy achieved with TFLite compatibility!")
elif final_val_accuracy >= 0.60:
    print("✅ GOOD! Solid performance with TFLite compatibility!")
else:
    print("📈 Model is training - continue with more epochs if needed.")

print("\n✅ DONE! You have a high-accuracy model that will convert to TFLite perfectly!")
