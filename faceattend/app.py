import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import uuid
import json
import zipfile
import shutil
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'faceattend_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faceattend.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['DATASET_FOLDER'] = 'dataset'
app.config['BACKUP_FOLDER'] = 'backups'

for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER'], 
              app.config['DATASET_FOLDER'], app.config['BACKUP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)


db = SQLAlchemy(app)


class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    face_images = db.relationship('FaceImage', backref='student', lazy=True, cascade="all, delete-orphan")

class FaceImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.now)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time_in = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), default='Present')
    created_at = db.Column(db.DateTime, default=datetime.now)
    student = db.relationship('Student', backref=db.backref('attendances', lazy=True))

class ModelTraining(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(20), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    params = db.Column(db.Text, nullable=False)  # JSON string of parameters
    trained_at = db.Column(db.DateTime, default=datetime.now)
    model_path = db.Column(db.String(200), nullable=False)
    is_active = db.Column(db.Boolean, default=False)

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setting_name = db.Column(db.String(100), unique=True, nullable=False)
    setting_value = db.Column(db.String(200), nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

class SystemBackup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    backup_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    backup_type = db.Column(db.String(20), default='auto')  # 'auto' or 'manual'

# Create all tables
with app.app_context():
    db.create_all()
    
    # Add default admin if none exists
    if not Admin.query.first():
        default_admin = Admin(
            username='admin',
            password=generate_password_hash('admin123'),
            email='admin@faceattend.com'
        )
        db.session.add(default_admin)
        
        # Add default settings
        default_settings = [
            SystemSettings(setting_name='camera_source', setting_value='Default Camera'),
            SystemSettings(setting_name='camera_resolution', setting_value='720p'),
            SystemSettings(setting_name='detection_threshold', setting_value='0.5'),
            SystemSettings(setting_name='class_start_time', setting_value='09:00'),
            SystemSettings(setting_name='class_end_time', setting_value='17:00'),
            SystemSettings(setting_name='late_threshold', setting_value='15'),
            SystemSettings(setting_name='email_notifications', setting_value='false')
        ]
        for setting in default_settings:
            db.session.add(setting)
        
        db.session.commit()

# Face detection function
def detect_face(image):
    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    # Process the largest face
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Expand the face area slightly
    expansion = int(0.1 * max(w, h))
    x = max(0, x - expansion)
    y = max(0, y - expansion)
    w = min(image.shape[1] - x, w + 2 * expansion)
    h = min(image.shape[0] - y, h + 2 * expansion)
    
    # Extract and return the face
    face = image[y:y+h, x:x+w]
    return cv2.resize(face, (160, 160))  # Resize to model input size

# Preprocess image for model input
def preprocess_image(image):
    # Normalize pixel values
    image = image.astype('float32')
    image = (image - 127.5) / 127.5
    return image

# Create and train the CNN model
def create_model(input_shape=(160, 160, 3), num_classes=None, learning_rate=0.001, 
                 optimizer_name='Adam', activation='relu', dropout_rate=0.5, num_conv_layers=4):
    
    # Create model base
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base model
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Configure optimizer
    if optimizer_name == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:  # RMSprop
        optimizer = RMSprop(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to train the model
def train_model(batch_size=32, epochs=20, learning_rate=0.001, validation_split=0.2, 
                optimizer_name='Adam', activation='relu', dropout_rate=0.5, num_conv_layers=4):
    
    # Setup data directories
    train_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    
    # Count number of classes (students)
    num_classes = len(os.listdir(train_dir))
    
    # Data augmentation
    data_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Create data generators
    train_generator = data_gen.flow_from_directory(
        train_dir,
        target_size=(160, 160),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = data_gen.flow_from_directory(
        train_dir,
        target_size=(160, 160),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create model
    model = create_model(
        num_classes=num_classes, 
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        activation=activation,
        dropout_rate=dropout_rate,
        num_conv_layers=num_conv_layers
    )
    
    # Callbacks
    model_version = f"v{len(os.listdir(app.config['MODEL_FOLDER'])) + 1}.0.0"
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"model_{model_version}.h5")
    
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=3, monitor='val_accuracy')
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(validation_generator)
    
    # Get class indices
    class_indices = train_generator.class_indices
    class_indices = {v: k for k, v in class_indices.items()}
    
    # Save class indices
    with open(os.path.join(app.config['MODEL_FOLDER'], f"class_indices_{model_version}.json"), 'w') as f:
        json.dump(class_indices, f)
    
    # Calculate precision, recall, F1 score
    y_true = validation_generator.classes
    y_pred = []
    
    # Get predictions
    for i in range(len(validation_generator)):
        batch_images, _ = validation_generator[i]
        batch_preds = model.predict(batch_images)
        y_pred.extend(np.argmax(batch_preds, axis=1))
    
    y_pred = y_pred[:len(y_true)]  # Make sure lengths match
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Save training history
    with open(os.path.join(app.config['MODEL_FOLDER'], f"history_{model_version}.json"), 'w') as f:
        json.dump({
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }, f)
    
    # Save model parameters
    params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'validation_split': validation_split,
        'optimizer': optimizer_name,
        'activation': activation,
        'dropout_rate': dropout_rate,
        'num_conv_layers': num_conv_layers
    }
    
    # Save model info to database
    new_model = ModelTraining(
        version=model_version,
        accuracy=float(val_accuracy),
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        params=json.dumps(params),
        model_path=model_path,
        is_active=True
    )
    
    # Deactivate other models
    with app.app_context():
        ModelTraining.query.update({ModelTraining.is_active: False})
        db.session.add(new_model)
        db.session.commit()
    
    return model, val_accuracy, precision, recall, f1, model_version

# Function to prepare dataset from uploaded images
def prepare_dataset():
    # Create dataset directory structure
    train_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    
    # Clear existing dataset
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    
    # Get all students
    students = Student.query.all()
    
    for student in students:
        # Create directory for student
        student_dir = os.path.join(train_dir, student.student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Copy face images to student directory
        for face_image in student.face_images:
            img_path = face_image.image_path
            
            if os.path.exists(img_path):
                # Read image
                img = cv2.imread(img_path)
                
                # Detect and crop face
                face = detect_face(img)
                
                if face is not None:
                    # Save processed face image
                    dest_path = os.path.join(student_dir, os.path.basename(img_path))
                    cv2.imwrite(dest_path, face)

# Function to apply data augmentation
def apply_augmentation(augmentation_factor=2, apply_rotation=True, apply_flip=True, 
                      apply_brightness=True, apply_zoom=True, apply_shift=True):
    
    train_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    
    # Ensure training directory exists
    if not os.path.exists(train_dir):
        prepare_dataset()
    
    # For each student directory
    for student_id in os.listdir(train_dir):
        student_dir = os.path.join(train_dir, student_id)
        
        # Get original images
        original_images = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in original_images:
            img_path = os.path.join(student_dir, img_file)
            img = cv2.imread(img_path)
            
            # Generate augmented images
            for i in range(augmentation_factor - 1):  # -1 because we already have the original
                augmented = img.copy()
                
                # Apply various augmentations
                if apply_rotation:
                    angle = np.random.uniform(-15, 15)
                    h, w = augmented.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    augmented = cv2.warpAffine(augmented, M, (w, h))
                
                if apply_flip and np.random.random() > 0.5:
                    augmented = cv2.flip(augmented, 1)  # horizontal flip
                
                if apply_brightness:
                    alpha = np.random.uniform(0.8, 1.2)  # brightness factor
                    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
                
                if apply_zoom:
                    h, w = augmented.shape[:2]
                    zoom = np.random.uniform(0.9, 1.1)
                    
                    # Calculate new dimensions
                    new_h, new_w = int(zoom * h), int(zoom * w)
                    
                    # Crop from center or pad
                    if zoom > 1:  # zoom in (crop)
                        y1 = max(0, int((new_h - h) / 2))
                        x1 = max(0, int((new_w - w) / 2))
                        augmented = cv2.resize(augmented, (new_w, new_h))
                        augmented = augmented[y1:y1+h, x1:x1+w]
                    else:  # zoom out (pad)
                        augmented = cv2.resize(augmented, (new_w, new_h))
                        y1 = max(0, int((h - new_h) / 2))
                        x1 = max(0, int((w - new_w) / 2))
                        
                        temp = np.zeros((h, w, 3), dtype=np.uint8)
                        temp[y1:y1+new_h, x1:x1+new_w] = augmented
                        augmented = temp
                
                if apply_shift:
                    h, w = augmented.shape[:2]
                    shift_x = int(w * np.random.uniform(-0.05, 0.05))
                    shift_y = int(h * np.random.uniform(-0.05, 0.05))
                    
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    augmented = cv2.warpAffine(augmented, M, (w, h))
                
                # Save augmented image
                base_name, ext = os.path.splitext(img_file)
                aug_img_path = os.path.join(student_dir, f"{base_name}_aug{i}{ext}")
                cv2.imwrite(aug_img_path, augmented)

# Function to create a system backup
def create_backup(backup_type='auto'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"backup_{timestamp}.zip"
    backup_path = os.path.join(app.config['BACKUP_FOLDER'], backup_filename)
    
    with zipfile.ZipFile(backup_path, 'w') as zipf:
        # Backup database
        zipf.write('faceattend.db', 'faceattend.db')
        
        # Backup models
        for root, _, files in os.walk(app.config['MODEL_FOLDER']):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(app.config['MODEL_FOLDER'])))
        
        # Backup student face images
        for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(app.config['UPLOAD_FOLDER'])))
    
    # Add backup record to database
    new_backup = SystemBackup(
        backup_path=backup_path,
        backup_type=backup_type
    )
    
    db.session.add(new_backup)
    db.session.commit()
    
    return backup_path

# Function to restore from backup
def restore_from_backup(backup_path):
    # Extract backup
    with zipfile.ZipFile(backup_path, 'r') as zipf:
        # Create a temporary directory
        temp_dir = os.path.join(app.config['BACKUP_FOLDER'], 'temp_restore')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Extract all files
        zipf.extractall(temp_dir)
        
        # Restore database
        db_path = os.path.join(temp_dir, 'faceattend.db')
        if os.path.exists(db_path):
            # Close current database connection
            db.session.close()
            
            # Copy restored database
            shutil.copy(db_path, 'faceattend.db')
        
        # Restore models
        model_dir = os.path.join(temp_dir, app.config['MODEL_FOLDER'])
        if os.path.exists(model_dir):
            # Clear current models
            shutil.rmtree(app.config['MODEL_FOLDER'], ignore_errors=True)
            os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
            
            # Copy restored models
            for item in os.listdir(model_dir):
                s = os.path.join(model_dir, item)
                d = os.path.join(app.config['MODEL_FOLDER'], item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, False, None)
                else:
                    shutil.copy2(s, d)
        
        # Restore student face images
        upload_dir = os.path.join(temp_dir, app.config['UPLOAD_FOLDER'])
        if os.path.exists(upload_dir):
            # Clear current uploads
            shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Copy restored uploads
            for item in os.listdir(upload_dir):
                s = os.path.join(upload_dir, item)
                d = os.path.join(app.config['UPLOAD_FOLDER'], item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, False, None)
                else:
                    shutil.copy2(s, d)
        
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True

# Function to recognize faces using the trained model
def recognize_face(face_image):
    # Get active model
    active_model = ModelTraining.query.filter_by(is_active=True).first()
    
    if not active_model:
        return None, 0.0
    
    # Load model
    model = load_model(active_model.model_path)
    
    # Load class indices
    class_indices_path = os.path.join(
        app.config['MODEL_FOLDER'], 
        f"class_indices_{active_model.version}.json"
    )
    
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Preprocess image
    face = cv2.resize(face_image, (160, 160))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    
    # Get prediction
    predictions = model.predict(face)[0]
    max_prob = np.max(predictions)
    pred_class = np.argmax(predictions)
    
    # Get student ID
    student_id = class_indices[str(pred_class)]
    
    return student_id, float(max_prob)

# Function to visualize feature maps
def visualize_feature_maps(model_version):
    # Load model
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"model_{model_version}.h5")
    model = load_model(model_path)
    
    # Create a model that outputs feature maps from a specific layer
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get a sample image
    sample_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    student_dirs = os.listdir(sample_dir)
    
    if not student_dirs:
        return None
    
    student_dir = os.path.join(sample_dir, student_dirs[0])
    images = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        return None
    
    sample_image_path = os.path.join(student_dir, images[0])
    img = cv2.imread(sample_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Get feature maps
    activations = activation_model.predict(img)
    
    # Create figure for visualization
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    # Plot original image
    axes[0].imshow(img[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot feature maps
    for i in range(min(7, len(activations))):
        feature_map = activations[i]
        # Plot first channel of each feature map
        axes[i+1].imshow(feature_map[0, :, :, 0], cmap='viridis')
        axes[i+1].set_title(f'Layer {i+1}')
        axes[i+1].axis('off')
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode buffer to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_str

# Function to generate t-SNE visualization
def generate_tsne_visualization(model_version):
    # Load model
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"model_{model_version}.h5")
    base_model = load_model(model_path)
    
    # Create a feature extraction model (without the classification head)
    feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    
    # Get class indices
    class_indices_path = os.path.join(app.config['MODEL_FOLDER'], f"class_indices_{model_version}.json")
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Process a subset of the dataset
    train_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    
    features = []
    labels = []
    
    # Limit to 10 images per student for better visualization
    for student_id in os.listdir(train_dir):
        student_dir = os.path.join(train_dir, student_id)
        images = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in images[:10]:  # Take only first 10 images
            img_path = os.path.join(student_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (160, 160))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Extract features
            feature_vector = feature_model.predict(img)
            features.append(feature_vector.flatten())
            labels.append(student_id)
            
            if len(features) >= 300:  # Limit to 300 points for faster computation
                break
        
        if len(features) >= 300:
            break
    
    if not features:
        return None
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(np.array(features))
    
    # Create a color map for different classes
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    
    # Plot t-SNE
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            features_2d[indices, 0], 
            features_2d[indices, 1], 
            color=color_map[label],
            label=label,
            alpha=0.7
        )
    
    plt.title('t-SNE Visualization of Face Embeddings')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode buffer to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# Generate confusion matrix
def generate_confusion_matrix(model_version):
    # Load model
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"model_{model_version}.h5")
    model = load_model(model_path)
    
    # Load class indices
    class_indices_path = os.path.join(app.config['MODEL_FOLDER'], f"class_indices_{model_version}.json")
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    
    # Setup validation data generator
    train_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    validation_generator = data_gen.flow_from_directory(
        train_dir,
        target_size=(160, 160),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Get predictions
    y_pred = model.predict(validation_generator)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create class labels
    labels = [class_indices[str(i)] for i in range(len(class_indices))]
    
    # Limit the number of classes for better visualization
    if len(labels) > 15:
        top_indices = np.argsort(np.sum(cm, axis=1))[-15:]
        cm = cm[top_indices][:, top_indices]
        labels = [labels[i] for i in top_indices]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    
    # Add count numbers
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode buffer to base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_str

# API Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    admin = Admin.query.filter_by(username=username).first()
    
    if admin and check_password_hash(admin.password, password):
        session['admin_id'] = admin.id
        return jsonify({'success': True, 'message': 'Login successful!'})
    
    return jsonify({'success': False, 'message': 'Invalid username or password.'})

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_id', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('index'))
    
    # Get dashboard data
    total_students = Student.query.count()
    
    today = datetime.now().date()
    present_today = Attendance.query.filter_by(date=today).count()
    
    # Get active model
    active_model = ModelTraining.query.filter_by(is_active=True).first()
    model_accuracy = active_model.accuracy if active_model else 0
    
    # Get recent attendance
    recent_attendance = (Attendance.query
                         .join(Student)
                         .with_entities(
                             Student.student_id,
                             Student.name,
                             Student.class_name,
                             Attendance.time_in,
                             Attendance.status
                         )
                         .order_by(Attendance.created_at.desc())
                         .limit(10)
                         .all())
    
    return render_template('dashboard.html', 
                          total_students=total_students,
                          present_today=present_today,
                          model_accuracy=model_accuracy,
                          recent_attendance=recent_attendance)

@app.route('/attendance/records')
def attendance_records():
    if 'admin_id' not in session:
        return redirect(url_for('index'))
    
    # Get filter parameters
    class_name = request.args.get('class', 'All Classes')
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    status = request.args.get('status', 'All Status')
    page = int(request.args.get('page', 1))
    
    # Build query
    query = Attendance.query.join(Student)
    
    if class_name != 'All Classes':
        query = query.filter(Student.class_name == class_name)
    
    if date_str:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Attendance.date == date)
        except ValueError:
            pass
    
    if status != 'All Status':
        query = query.filter(Attendance.status == status)
    
    # Paginate results
    per_page = 10
    pagination = query.paginate(page=page, per_page=per_page)
    
    attendance_records = []
    for att in pagination.items:
        student = Student.query.get(att.student_id)
        attendance_records.append({
            'student_id': student.student_id,
            'name': student.name,
            'class': student.class_name,
            'date': att.date.strftime('%Y-%m-%d'),
            'time_in': att.time_in.strftime('%H:%M:%S'),
            'status': att.status
        })
    
    # Get unique class names
    classes = [row[0] for row in db.session.query(Student.class_name).distinct().all()]
    
    return render_template('attendance_records.html',
                          attendance_records=attendance_records,
                          classes=classes,
                          pagination=pagination)

@app.route('/model/training')
def model_training():
    if 'admin_id' not in session:
        return redirect(url_for('index'))
    
    # Get active model
    active_model = ModelTraining.query.filter_by(is_active=True).first()
    
    # Get training history if available
    training_history = None
    if active_model:
        history_path = os.path.join(app.config['MODEL_FOLDER'], f"history_{active_model.version}.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_history = json.load(f)
        
        # Parse parameters
        model_params = json.loads(active_model.params)
    else:
        model_params = {
            'batch_size': 32,
            'epochs': 20,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'optimizer': 'Adam',
            'activation': 'relu',
            'dropout_rate': 0.5,
            'num_conv_layers': 4
        }
    
    # Count dataset size
    dataset_size = 0
    num_students = 0
    
    train_dir = os.path.join(app.config['DATASET_FOLDER'], 'train')
    if os.path.exists(train_dir):
        num_students = len(os.listdir(train_dir))
        
        for student_dir in os.listdir(train_dir):
            student_path = os.path.join(train_dir, student_dir)
            if os.path.isdir(student_path):
                dataset_size += len([f for f in os.listdir(student_path) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    return render_template('model_training.html',
                          active_model=active_model,
                          training_history=training_history,
                          model_params=model_params,
                          dataset_size=dataset_size,
                          num_students=num_students)

@app.route('/student/management')
def student_management():
    if 'admin_id' not in session:
        return redirect(url_for('index'))
    
    # Get filter parameters
    page = int(request.args.get('page', 1))
    
    # Paginate results
    per_page = 10
    pagination = Student.query.paginate(page=page, per_page=per_page)
    
    students = []
    for student in pagination.items:
        face_images = [face.image_path for face in student.face_images]
        students.append({
            'id': student.id,
            'student_id': student.student_id,
            'name': student.name,
            'class': student.class_name,
            'email': student.email,
            'face_images': face_images
        })
    
    return render_template('student_management.html',
                          students=students,
                          pagination=pagination)

@app.route('/model/evaluation')
def model_evaluation():
    if 'admin_id' not in session:
        return redirect(url_for('index'))
    
    # Get active model
    active_model = ModelTraining.query.filter_by(is_active=True).first()
    
    if not active_model:
        return render_template('model_evaluation.html',
                              active_model=None,
                              confusion_matrix=None,
                              feature_maps=None,
                              tsne_visualization=None)
    
    # Generate visualizations
    confusion_matrix_img = generate_confusion_matrix(active_model.version)
    feature_maps_img = visualize_feature_maps(active_model.version)
    tsne_visualization_img = generate_tsne_visualization(active_model.version)
    
    return render_template('model_evaluation.html',
                          active_model=active_model,
                          confusion_matrix=confusion_matrix_img,
                          feature_maps=feature_maps_img,
                          tsne_visualization=tsne_visualization_img)

@app.route('/system/settings')
def system_settings():
    if 'admin_id' not in session:
        return redirect(url_for('index'))
    
    # Get settings
    settings = {}
    for setting in SystemSettings.query.all():
        settings[setting.setting_name] = setting.setting_value
    
    # Get admin info
    admin = Admin.query.get(session['admin_id'])
    
    # Get latest backup
    latest_backup = SystemBackup.query.order_by(SystemBackup.created_at.desc()).first()
    
    return render_template('system_settings.html',
                          settings=settings,
                          admin=admin,
                          latest_backup=latest_backup)

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')

# API endpoints
@app.route('/api/admin/update', methods=['POST'])
def api_admin_update():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    email = data.get('email')
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    admin = Admin.query.get(session['admin_id'])
    
    if not check_password_hash(admin.password, current_password):
        return jsonify({'success': False, 'message': 'Current password is incorrect'})
    
    admin.email = email
    
    if new_password:
        admin.password = generate_password_hash(new_password)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Admin account updated'})

@app.route('/api/settings/update', methods=['POST'])
def api_settings_update():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    setting_type = data.get('type')
    
    if setting_type == 'camera':
        # Update camera settings
        camera_source = data.get('camera_source')
        resolution = data.get('resolution')
        threshold = data.get('threshold')
        
        SystemSettings.query.filter_by(setting_name='camera_source').update({'setting_value': camera_source})
        SystemSettings.query.filter_by(setting_name='camera_resolution').update({'setting_value': resolution})
        SystemSettings.query.filter_by(setting_name='detection_threshold').update({'setting_value': str(threshold)})
    
    elif setting_type == 'attendance':
        # Update attendance settings
        class_start = data.get('class_start')
        class_end = data.get('class_end')
        late_threshold = data.get('late_threshold')
        email_notifications = data.get('email_notifications', False)
        
        SystemSettings.query.filter_by(setting_name='class_start_time').update({'setting_value': class_start})
        SystemSettings.query.filter_by(setting_name='class_end_time').update({'setting_value': class_end})
        SystemSettings.query.filter_by(setting_name='late_threshold').update({'setting_value': str(late_threshold)})
        SystemSettings.query.filter_by(setting_name='email_notifications').update(
            {'setting_value': 'true' if email_notifications else 'false'}
        )
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Settings updated'})

@app.route('/api/backup/create', methods=['POST'])
def api_backup_create():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    try:
        backup_path = create_backup('manual')
        return jsonify({
            'success': True, 
            'message': 'Backup created successfully',
            'backup_path': backup_path,
            'created_at': datetime.now().strftime('%B %d, %Y at %I:%M %p')
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error creating backup: {str(e)}'})

@app.route('/api/backup/schedule', methods=['POST'])
def api_backup_schedule():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    frequency = data.get('frequency')
    day = data.get('day')
    time = data.get('time')
    send_email = data.get('send_email', False)
    
    # Store backup schedule in settings
    SystemSettings.query.filter_by(setting_name='backup_frequency').update(
        {'setting_value': frequency}
    )
    SystemSettings.query.filter_by(setting_name='backup_day').update(
        {'setting_value': day}
    )
    SystemSettings.query.filter_by(setting_name='backup_time').update(
        {'setting_value': time}
    )
    SystemSettings.query.filter_by(setting_name='backup_email').update(
        {'setting_value': 'true' if send_email else 'false'}
    )
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Backup schedule updated'})

@app.route('/api/backup/restore', methods=['POST'])
def api_backup_restore():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    backup_source = data.get('backup_source')
    
    try:
        if backup_source.startswith('Automated') or backup_source.startswith('Manual'):
            # Get backup from database
            date_str = backup_source.split(' - ')[1].split(' (')[0]
            time_str = backup_source.split('(')[1].split(')')[0]
            
            backup_datetime = datetime.strptime(f"{date_str} {time_str}", "%b %d, %Y %I:%M %p")
            
            # Find closest backup
            backup = SystemBackup.query.filter(
                SystemBackup.created_at >= backup_datetime - timedelta(minutes=5),
                SystemBackup.created_at <= backup_datetime + timedelta(minutes=5)
            ).first()
            
            if not backup:
                return jsonify({'success': False, 'message': 'Backup not found'})
            
            backup_path = backup.backup_path
        else:
            # Handle uploaded backup file
            if 'backup_file' not in request.files:
                return jsonify({'success': False, 'message': 'No backup file provided'})
            
            backup_file = request.files['backup_file']
            backup_path = os.path.join(app.config['BACKUP_FOLDER'], f"uploaded_{secure_filename(backup_file.filename)}")
            backup_file.save(backup_path)
        
        # Restore from backup
        success = restore_from_backup(backup_path)
        
        if success:
            return jsonify({'success': True, 'message': 'System restored successfully'})
        else:
            return jsonify({'success': False, 'message': 'Error restoring system'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error restoring system: {str(e)}'})

@app.route('/api/student/add', methods=['POST'])
def api_student_add():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    student_id = request.form.get('student_id')
    name = request.form.get('name')
    class_name = request.form.get('class')
    email = request.form.get('email')
    
    # Check if student ID already exists
    if Student.query.filter_by(student_id=student_id).first():
        return jsonify({'success': False, 'message': 'Student ID already exists'})
    
    # Create new student
    new_student = Student(
        student_id=student_id,
        name=name,
        class_name=class_name,
        email=email
    )
    
    db.session.add(new_student)
    db.session.commit()
    
    # Process face images
    if 'face_images' in request.files:
        face_images = request.files.getlist('face_images')
        
        for image in face_images:
            if image and image.filename:
                # Save image
                filename = secure_filename(f"{student_id}_{uuid.uuid4()}{os.path.splitext(image.filename)[1]}")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                
                # Create face image record
                face_image = FaceImage(
                    student_id=new_student.id,
                    image_path=image_path
                )
                
                db.session.add(face_image)
        
        db.session.commit()
    
    return jsonify({'success': True, 'message': 'Student added successfully'})

@app.route('/api/student/edit', methods=['POST'])
def api_student_edit():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    student_id = request.form.get('student_id')
    name = request.form.get('name')
    class_name = request.form.get('class')
    email = request.form.get('email')
    student_db_id = request.form.get('id')
    
    # Get student
    student = Student.query.get(student_db_id)
    
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'})
    
    # Check if student ID already exists for another student
    existing = Student.query.filter_by(student_id=student_id).first()
    if existing and existing.id != int(student_db_id):
        return jsonify({'success': False, 'message': 'Student ID already exists'})
    
    # Update student
    student.student_id = student_id
    student.name = name
    student.class_name = class_name
    student.email = email
    
    # Process new face images
    if 'face_images' in request.files:
        face_images = request.files.getlist('face_images')
        
        for image in face_images:
            if image and image.filename:
                # Save image
                filename = secure_filename(f"{student_id}_{uuid.uuid4()}{os.path.splitext(image.filename)[1]}")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                
                # Create face image record
                face_image = FaceImage(
                    student_id=student.id,
                    image_path=image_path
                )
                
                db.session.add(face_image)
    
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Student updated successfully'})

@app.route('/api/student/delete', methods=['POST'])
def api_student_delete():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    student_id = data.get('student_id')
    
    # Get student
    student = Student.query.get(student_id)
    
    if not student:
        return jsonify({'success': False, 'message': 'Student not found'})
    
    # Delete face images
    for face_image in student.face_images:
        if os.path.exists(face_image.image_path):
            os.remove(face_image.image_path)
    
    # Delete student
    db.session.delete(student)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Student deleted successfully'})

@app.route('/api/student/import', methods=['POST'])
def api_student_import():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    if 'csv_file' not in request.files:
        return jsonify({'success': False, 'message': 'No CSV file provided'})
    
    csv_file = request.files['csv_file']
    
    if csv_file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_columns = ['student_id', 'name', 'class', 'email']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({'success': False, 'message': f'Missing required column: {col}'})
        
        # Import students
        imported_count = 0
        existing_count = 0
        
        for _, row in df.iterrows():
            # Check if student already exists
            if Student.query.filter_by(student_id=row['student_id']).first():
                existing_count += 1
                continue
            
            # Create new student
            new_student = Student(
                student_id=row['student_id'],
                name=row['name'],
                class_name=row['class'],
                email=row['email']
            )
            
            db.session.add(new_student)
            imported_count += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': f'Imported {imported_count} students. {existing_count} students already existed.'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error importing students: {str(e)}'})

@app.route('/api/student/export', methods=['GET'])
def api_student_export():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    # Get all students
    students = Student.query.all()
    
    # Create dataframe
    data = []
    for student in students:
        data.append({
            'student_id': student.student_id,
            'name': student.name,
            'class': student.class_name,
            'email': student.email,
            'created_at': student.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df = pd.DataFrame(data)
    
    # Create CSV file
    csv_file = os.path.join(app.config['UPLOAD_FOLDER'], 'students_export.csv')
    df.to_csv(csv_file, index=False)
    
    return send_file(csv_file, as_attachment=True, download_name='students_export.csv')

@app.route('/api/attendance/export', methods=['GET'])
def api_attendance_export():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    # Get filter parameters
    class_name = request.args.get('class', 'All Classes')
    date_str = request.args.get('date', '')
    status = request.args.get('status', 'All Status')
    export_format = request.args.get('format', 'csv')
    
    # Build query
    query = Attendance.query.join(Student)
    
    if class_name != 'All Classes':
        query = query.filter(Student.class_name == class_name)
    
    if date_str:
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            query = query.filter(Attendance.date == date)
        except ValueError:
            pass
    
    if status != 'All Status':
        query = query.filter(Attendance.status == status)
    
    # Get attendance records
    attendance_records = []
    for att in query.all():
        student = Student.query.get(att.student_id)
        attendance_records.append({
            'student_id': student.student_id,
            'name': student.name,
            'class': student.class_name,
            'date': att.date.strftime('%Y-%m-%d'),
            'time_in': att.time_in.strftime('%H:%M:%S'),
            'status': att.status
        })
    
    df = pd.DataFrame(attendance_records)
    
    if export_format == 'csv':
        # Create CSV file
        export_file = os.path.join(app.config['UPLOAD_FOLDER'], 'attendance_export.csv')
        df.to_csv(export_file, index=False)
        return send_file(export_file, as_attachment=True, download_name='attendance_export.csv')
    else:  # PDF
        # Create PDF file (simplified for this example)
        export_file = os.path.join(app.config['UPLOAD_FOLDER'], 'attendance_export.csv')
        df.to_csv(export_file, index=False)
        return send_file(export_file, as_attachment=True, download_name='attendance_export.pdf')

@app.route('/api/dataset/prepare', methods=['POST'])
def api_dataset_prepare():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    try:
        prepare_dataset()
        return jsonify({'success': True, 'message': 'Dataset prepared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error preparing dataset: {str(e)}'})

@app.route('/api/dataset/augment', methods=['POST'])
def api_dataset_augment():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    augmentation_factor = int(data.get('augmentation_factor', 2))
    rotation = data.get('rotation', True)
    flip = data.get('flip', True)
    brightness = data.get('brightness', True)
    zoom = data.get('zoom', True)
    shift = data.get('shift', True)
    
    try:
        apply_augmentation(
            augmentation_factor=augmentation_factor,
            apply_rotation=rotation,
            apply_flip=flip,
            apply_brightness=brightness,
            apply_zoom=zoom,
            apply_shift=shift
        )
        
        return jsonify({'success': True, 'message': 'Data augmentation applied successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error applying data augmentation: {str(e)}'})

@app.route('/api/model/train', methods=['POST'])
def api_model_train():
    if 'admin_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    data = request.get_json()
    
    # Get training parameters
    batch_size = int(data.get('batch_size', 32))
    epochs = int(data.get('epochs', 20))
    learning_rate = float(data.get('learning_rate', 0.001))
    validation_split = float(data.get('validation_split', 0.2))
    optimizer_name = data.get('optimizer', 'Adam')
    activation = data.get('activation', 'relu')
    dropout_rate = float(data.get('dropout_rate', 0.5))
    num_conv_layers = int(data.get('num_conv_layers', 4))
    
    try:
        # First prepare the dataset if not already done
        prepare_dataset()
        
        # Train model
        _, accuracy, precision, recall, f1, model_version = train_model(
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_split=validation_split,
            optimizer_name=optimizer_name,
            activation=activation,
            dropout_rate=dropout_rate,
            num_conv_layers=num_conv_layers
        )
        
        return jsonify({
            'success': True, 
            'message': 'Model trained successfully',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'model_version': model_version
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error training model: {str(e)}'})

@app.route('/api/recognition', methods=['POST'])
def api_recognition():
    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    try:
        # Read image
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Detect face
        face = detect_face(img)
        
        if face is None:
            return jsonify({'success': False, 'message': 'No face detected'})
        
        # Recognize face
        student_id, confidence = recognize_face(face)
        
        # Get detection threshold from settings
        threshold_setting = SystemSettings.query.filter_by(setting_name='detection_threshold').first()
        threshold = float(threshold_setting.setting_value if threshold_setting else 0.5)
        
        if confidence < threshold:
            return jsonify({'success': False, 'message': 'Face not recognized with sufficient confidence'})
        
        # Get student
        student = Student.query.filter_by(student_id=student_id).first()
        
        if not student:
            return jsonify({'success': False, 'message': 'Student not found in database'})
        
        # Record attendance
        today = datetime.now().date()
        now = datetime.now().time()
        
        # Check if student already has attendance for today
        existing_attendance = Attendance.query.filter_by(
            student_id=student.id,
            date=today
        ).first()
        
        if existing_attendance:
            return jsonify({
                'success': True,
                'message': 'Attendance already recorded for today',
                'student': {
                    'id': student.student_id,
                    'name': student.name,
                    'class': student.class_name
                },
                'attendance': {
                    'date': today.strftime('%Y-%m-%d'),
                    'time': existing_attendance.time_in.strftime('%H:%M:%S'),
                    'status': existing_attendance.status
                }
            })
        
        # Get class hours settings
        start_time_setting = SystemSettings.query.filter_by(setting_name='class_start_time').first()
        late_threshold_setting = SystemSettings.query.filter_by(setting_name='late_threshold').first()
        
        start_time_str = start_time_setting.setting_value if start_time_setting else '09:00'
        late_threshold = int(late_threshold_setting.setting_value if late_threshold_setting else 15)
        
        # Parse start time
        start_time = datetime.strptime(start_time_str, '%H:%M').time()
        
        # Calculate late threshold time
        start_datetime = datetime.combine(today, start_time)
        late_datetime = start_datetime + timedelta(minutes=late_threshold)
        late_time = late_datetime.time()
        
        # Determine status
        status = 'Present'
        if now > late_time:
            status = 'Late'
        
        # Record attendance
        new_attendance = Attendance(
            student_id=student.id,
            date=today,
            time_in=now,
            status=status
        )
        
        db.session.add(new_attendance)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Attendance recorded successfully',
            'student': {
                'id': student.student_id,
                'name': student.name,
                'class': student.class_name
            },
            'attendance': {
                'date': today.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'status': status
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error processing recognition: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)