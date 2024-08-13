# CNN model


# Importing the libraries needed
import numpy as np # mathematical operations including arrays and matrices
import matplotlib.pyplot as plt # plotting library for the Python 
import seaborn as sns # data visualization library based on matplotlib
from keras.models import Sequential # Sequential model is a linear stack of layers where each layer has exactly one input tensor and one output tensor 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # Convolutional, MaxPooling, Flatten, Dense, and Dropout layers from Keras API 
from keras.preprocessing.image import ImageDataGenerator 
from keras.optimizers import Adam # Adam optimizer
from sklearn.metrics import classification_report, confusion_matrix # Classification report and confusion matrix


# Image dimensions
img_width, img_height = 150, 150  # Adjusted for consistency

# Directory paths
train_data_dir = 
test_data_dir = 

# Parameters
epochs = 15 
batch_size = 32 
num_classes = 4 

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255) # Rescale the pixel values to be between 0 and 1 
test_datagen = ImageDataGenerator(rescale=1. / 255) 

# Training set
training_set = train_datagen.flow_from_directory( #flow_from_directory() method is used to generate batches of image data directly from a directory
    train_data_dir,                               #Directory path
    target_size=(img_width, img_height),          #Size of the images
    batch_size=batch_size,                        #Size of the batches of data
    class_mode='categorical')                     #Type of label arrays that are returned

# Test set
testing_set = test_datagen.flow_from_directory(  
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False) #Shuffling is set to False to ensure that the predictions are in the same order as the test data

# CNN model
model = Sequential() #Sequential model with layers stacked on top of each other 

#Adding Layers 
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu')) #Convolutional layer with 32 filters, kernel size of 3x3, input shape of 150x150x3, and ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2))) #Max pooling layer with pool size of 2x2

model.add(Conv2D(32, (3, 3), activation='relu')) #Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2))) #Max pooling layer with pool size of 2x2

model.add(Conv2D(64, (3, 3), activation='relu')) #Convolutional layer with 64 filters, kernel size of 3x3, and ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2))) #Max pooling layer with pool size of 2x2

model.add(Conv2D(64, (3, 3), activation='relu')) #Convolutional layer with 64 filters, kernel size of 3x3, and ReLU activation function
model.add(MaxPooling2D(pool_size=(2, 2))) #Max pooling layer with pool size of 2x2

model.add(Flatten()) #Flatten layer

# Determine the correct number of units for the Dense layer
model.add(Dense(128, activation='relu')) #Dense layer with 128 units and ReLU activation function

# Add Dropout layer
model.add(Dropout(0.5)) #Dropout layer with a rate of 0.5

model.add(Dense(num_classes, activation='softmax')) #Dense layer with 4 units and softmax activation function

# Compile the model
model.compile(optimizer=Adam(),  # Optimizer
              loss='categorical_crossentropy', # Loss function
              metrics=['accuracy'])  # Metric to evaluate

# Train the model
history = model.fit(  # Fit the model
    training_set,  # Training data
    steps_per_epoch=training_set.samples // training_set.batch_size,  # Number of steps to complete one epoch
    epochs=epochs,  # Number of epochs
    validation_data=testing_set,   # Validation data
    validation_steps=testing_set.samples // testing_set.batch_size) # Number of steps to complete one epoch

# Calculate steps per epoch for testing
test_steps_per_epoch = np.math.ceil(testing_set.samples / testing_set.batch_size) 

# Predict classes
predictions = model.predict(testing_set, steps=test_steps_per_epoch) # Predict the classes
predicted_classes = np.argmax(predictions, axis=1) # Predicted classes
true_classes = testing_set.classes # True classes
class_labels = list(testing_set.class_indices.keys()) # Class labels

# Print classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels) # Classification report
print(report)
 
# Evaluate the model
test_loss, test_accuracy = model.evaluate(testing_set, steps=testing_set.samples // testing_set.batch_size) # Evaluate the model
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Print summary of the model parameters
model.summary()

# Plotting accuracy and loss over epochs
plt.figure(figsize=(12, 4)) # Figure size

# Accuracy plot
plt.subplot(1, 2, 1) # Subplot for accuracy 
plt.plot(history.history['accuracy']) # Plot accuracy
plt.plot(history.history['val_accuracy']) # Plot validation accuracy
plt.title('Model accuracy') # Title
plt.xlabel('Epoch') # X-axis label
plt.ylabel('Accuracy') # Y-axis label
plt.legend(['Train', 'Validation'], loc='upper left') # Key

# Loss plot
plt.subplot(1, 2, 2) # Subplot for loss 
plt.plot(history.history['loss']) # Plot loss
plt.plot(history.history['val_loss']) # Plot validation loss
plt.title('Model loss') # Title
plt.xlabel('Epoch') # X-axis label
plt.ylabel('Loss') # Y-axis label
plt.legend(['Train', 'Validation'], loc='upper left') # Key

plt.show() # Show the plot

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes) # Confusion matrix
plt.figure(figsize=(8, 6)) # Figure size
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels) # Heatmap
plt.title('Confusion Matrix') # Title
plt.xlabel('Predicted Label') # X-axis label
plt.ylabel('True Label') # Y-axis label
plt.show() # Show the plot
