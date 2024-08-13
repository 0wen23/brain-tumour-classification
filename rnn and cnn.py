#rnncnn


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Image dimensions
img_width, img_height = 150, 150
num_classes = 4
epochs = 15
batch_size = 32

# Directory paths
train_data_dir = 
test_data_dir = 

# Data generators
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Training set
training_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Test set
testing_set = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

#CNN-RNN Model
model = Sequential() # Initialize the model

# CNN layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3))) # 3 for RGB channels 
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling layer with 2x2 pool size

model.add(Conv2D(64, (3, 3), activation='relu')) # 64 filters with 3x3 kernel size
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling layer with 2x2 pool size

model.add(Conv2D(128, (3, 3), activation='relu')) # 128 filters with 3x3 kernel size 
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling layer with 2x2 pool size 

model.add(Flatten()) # Flatten the output of the convolutional layers to pass to the RNN

# Reshape to fit RNN input
model.add(Reshape((1, -1)))  # Reshape to (timesteps, features)

# Add RNN layer
model.add(LSTM(50, return_sequences=False))  #adjust

# Add Dense Layers
model.add(Dense(128, activation='relu')) # 128 neurons in the hidden layer with ReLU activation function 
model.add(Dropout(0.7)) # Dropout layer to prevent overfitting 
model.add(Dense(num_classes, activation='softmax')) # Output layer with softmax activation function

# Compile the model
model.compile(optimizer=Adam(),  # Adam optimizer with default learning rate 
              loss='categorical_crossentropy', # Loss function for classes > 2 
              metrics=['accuracy']) # Metric to evaluate the model 

# Train the model
history = model.fit(
    training_set,
    steps_per_epoch=training_set.samples // training_set.batch_size,
    epochs=epochs,
    validation_data=testing_set,
    validation_steps=testing_set.samples // testing_set.batch_size)

# Calculate steps per epoch for testing
test_steps_per_epoch = np.math.ceil(testing_set.samples / testing_set.batch_size)

# Predict classes
predictions = model.predict(testing_set, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = testing_set.classes
class_labels = list(testing_set.class_indices.keys())

# Print classification report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(testing_set, steps=test_steps_per_epoch)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Print summary of the model parameters
model.summary()

# Plotting accuracy and loss over epochs
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
