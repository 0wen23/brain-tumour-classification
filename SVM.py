# SVM model

import os
import numpy as np
import cv2  # OpenCV
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Import completed.")

# Directory paths
data_dir = 

# Image dimensions
img_height, img_width = 150, 150

# Load images and labels
def load_images(data_dir): # Function to load images and labels
    images = [] # List to store images
    labels = [] # List to store labels
    for label in os.listdir(data_dir): # List all directories in the data directory
        for file in os.listdir(os.path.join(data_dir, label)): # List all files in the directory
            img_path = os.path.join(data_dir, label, file) # Image path
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            img = cv2.resize(img, (img_height, img_width)) # Resize
            images.append(img) # Append image to the list
            labels.append(label) # Append label to the list
    return np.array(images), np.array(labels) # Return images and labels as numpy arrays

print("Loading training data...")
# Load train and test data
train_images, train_labels = load_images(os.path.join(data_dir, 'Training'))
print(f"Training data loaded: {len(train_images)} images.")

print("/nLoading testing data...")
test_images, test_labels = load_images(os.path.join(data_dir, 'Testing'))
print(f"Testing data loaded: {len(test_images)} images.")

# Encode labels
label_encoder = LabelEncoder() # Label encoder
train_labels_encoded = label_encoder.fit_transform(train_labels) # Encode training labels
test_labels_encoded = label_encoder.transform(test_labels) # Encode testing labels
print("Labels encoded.") # Print 

# Extract HOG features
def extract_hog_features(images): # Function to extract HOG features
    hog_features = [] # List to store HOG features
    for image in images: # Loop through images
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False) # Extract HOG features
        hog_features.append(features) # Append features to the list
    return np.array(hog_features) # Return HOG features as numpy array

print("Extracting HOG features for training data...")
train_features = extract_hog_features(train_images)
print("HOG features extracted for training data.")

print("Extracting HOG features for testing data...")
test_features = extract_hog_features(test_images)
print("HOG features extracted for testing data.")

# Standardise the features
scaler = StandardScaler() # Standard scaler
train_features = scaler.fit_transform(train_features) # Standardise training features
test_features = scaler.transform(test_features) # Standardise testing features
print("Features standardised.")

# Train the SVM
print("Training SVM...") # Print notice
svm = SVC(kernel='linear', probability=True) # SVM model
svm.fit(train_features, train_labels_encoded) # Train the model
print("SVM training completed.") # Print notice of completion

# Predict the labels on the test set
print("Predicting on test set...") # Print notice
test_predictions = svm.predict(test_features) # Predict labels on the test set
print("Prediction completed.") 

# Evaluate the model
test_accuracy = accuracy_score(test_labels_encoded, test_predictions) # Test accuracy
print(f'Test accuracy: {test_accuracy * 100:.2f}%') # Print test accuracy

# Classification report
report = classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_) # Classification report
print(report) # Print classification report

# Confusion matrix
cm = confusion_matrix(test_labels_encoded, test_predictions) # Confusion matrix
plt.figure(figsize=(8, 6)) # Figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_) # Plot confusion matrix
plt.title('Confusion Matrix') # Title
plt.xlabel('Predicted labels') # X-axis label
plt.ylabel('True labels') # Y-axis label
plt.show() 