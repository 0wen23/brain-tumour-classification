#xgb 

import os
import numpy as np
import cv2  # OpenCV
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Directory paths
data_dir = 

# Image dimensions
img_height, img_width = 150, 150

# Load images and labels
def load_images(data_dir): 
    images = [] #empty list for images
    labels = [] #empty list for labels
    for label in os.listdir(data_dir):  #iterating through the directory
        for file in os.listdir(os.path.join(data_dir, label)):  #iterating through the directory
            img_path = os.path.join(data_dir, label, file)  #joining the path of the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  #reading the image in grayscale
            img = cv2.resize(img, (img_height, img_width))  #resizing the image
            images.append(img)  #appending the image to the list
            labels.append(label)  #appending the label to the list
    return np.array(images), np.array(labels)  #returning the images and labels

print("Loading training data...")  #printing the stage of loading the training data
train_images, train_labels = load_images(os.path.join(data_dir, 'Training'))  #loading the training data
print(f"Training data loaded: {len(train_images)} images.")  #printing the number of images loaded

print("Loading testing data...")  #printing the stage of loading the testing data
test_images, test_labels = load_images(os.path.join(data_dir, 'Testing'))  #loading the testing data
print(f"Testing data loaded: {len(test_images)} images.")  #printing the number of images loaded

# Encode labels
label_encoder = LabelEncoder()  #encoding the labels using the label encoder function from sklearn library 
train_labels_encoded = label_encoder.fit_transform(train_labels)  #fitting and transforming the training labels 
test_labels_encoded = label_encoder.transform(test_labels) #transforming the testing labels 
print("Labels encoded.")  #printing the completion of encoding the labels

# Extract HOG features
def extract_hog_features(images): #function to extract the HOG features
    hog_features = []   #empty list for HOG features
    for image in images:   #iterating through the images
        features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)   #extracting the HOG features from the image 
        hog_features.append(features) #appending the features to the list  
    return np.array(hog_features) #returning the HOG features

print("Extracting HOG features for training data...") #printing the stage of extracting the HOG features for training data 
train_features = extract_hog_features(train_images)   #extracting the HOG features for training data 
print("HOG features extracted for training data.")    #printing the completion of extracting the HOG features for training data 

print("Extracting HOG features for testing data...")  #printing the stage of extracting the HOG features for testing data
test_features = extract_hog_features(test_images)     #extracting the HOG features for testing data
print("HOG features extracted for testing data.")     #printing the completion of extracting the HOG features for testing data 

# Standardise the features 
scaler = StandardScaler()   #standardising the features using the StandardScaler function from the sklearn library 
train_features = scaler.fit_transform(train_features) #fitting and transforming the training features 
test_features = scaler.transform(test_features)  #transforming the testing features 
print("Features standardized.")  

# Train the XGBoost model
print("Training XGBoost model...") #printing the stages of the model training
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(train_labels_encoded)), eval_metric='mlogloss') #using the softmax function
xgb_model.fit(train_features, train_labels_encoded) #fitting the model
print("XGBoost model training completed.") #printing the completion of the model training

# Predict the labels on the test set
print("Predicting on test set...") 
test_predictions = xgb_model.predict(test_features)   #predicting the labels on the test set 
print("Prediction completed.")

# Evaluate the model
test_accuracy = accuracy_score(test_labels_encoded, test_predictions) #calculating the accuracy of the model
print(f'Test accuracy: {test_accuracy * 100:.2f}%')  #printing the accuracy of the model

# Classification report
report = classification_report(test_labels_encoded, test_predictions, target_names=label_encoder.classes_) # forming the classification report 
print(report)   #printing the classification report
 
# Confusion matrix
cm = confusion_matrix(test_labels_encoded, test_predictions) #forming the confusion matrix
plt.figure(figsize=(8, 6)) #plotting the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_) #plotting the heatmap
plt.title('Confusion Matrix') #title of the plot
plt.xlabel('Predicted labels') #x-axis label
plt.ylabel('True labels') #y-axis label
plt.show() #displaying the plot made
