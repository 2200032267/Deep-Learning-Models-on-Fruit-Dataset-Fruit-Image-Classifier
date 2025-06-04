import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Paths
dataset_dir = r'C:\PYTHON PY-CHARM\pythonProject1\dlproject\images'
train_dir = r'C:\PYTHON PY-CHARM\pythonProject1\dlproject\training'
val_dir = r'C:\PYTHON PY-CHARM\pythonProject1\dlproject\validation'

# Create train and validation folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split images into training and validation sets
for fruit_class in os.listdir(dataset_dir):
    fruit_class_path = os.path.join(dataset_dir, fruit_class)

    if os.path.isdir(fruit_class_path):
        os.makedirs(os.path.join(train_dir, fruit_class), exist_ok=True)
        os.makedirs(os.path.join(val_dir, fruit_class), exist_ok=True)

        image_files = [f for f in os.listdir(fruit_class_path) if os.path.isfile(os.path.join(fruit_class_path, f))]
        random.shuffle(image_files)

        split_index = int(0.8 * len(image_files))

        for i, image in enumerate(image_files):
            src_path = os.path.join(fruit_class_path, image)
            if i < split_index:
                dst_path = os.path.join(train_dir, fruit_class, image)
            else:
                dst_path = os.path.join(val_dir, fruit_class, image)

            shutil.copy(src_path, dst_path)

        print(f"Finished splitting {fruit_class} images into train and validation.")

# Image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important: Don't shuffle validation for correct evaluation
)

# Model building
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator
)

# Save the model
model.save('fruit_classification_model.h5')

# Model evaluation
val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation Accuracy: {val_acc:.4f}')

# Plotting training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predictions
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Classification Report
target_names = list(val_generator.class_indices.keys())
report_dict = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)

# Metrics DataFrame
metrics_df = pd.DataFrame({
    'Precision': [report_dict[label]['precision'] for label in target_names],
    'Recall': [report_dict[label]['recall'] for label in target_names],
    'F1-Score': [report_dict[label]['f1-score'] for label in target_names]
}, index=target_names)

# Bar Plot for metrics
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Class-wise Metrics (Precision, Recall, F1-Score)')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of Predicted Classes
plt.figure(figsize=(8, 6))
plt.hist(y_pred_classes, bins=np.arange(len(target_names) + 1) - 0.5, edgecolor='black')
plt.title('Distribution of Predicted Classes')
plt.xlabel('Class Labels')
plt.ylabel('Frequency')
plt.xticks(np.arange(len(target_names)), target_names, rotation=45)
plt.tight_layout()
plt.show()

# Scatter Plot of True vs Predicted Classes
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred_classes, alpha=0.5)
plt.title('Scatter Plot of True vs Predicted Classes')
plt.xlabel('True Classes')
plt.ylabel('Predicted Classes')
plt.xticks(np.arange(len(target_names)), target_names, rotation=45)
plt.tight_layout()
plt.show()

# Final Print Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=target_names))
