import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt

# 1. Load the dataset (CIFAR-10)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. Normalize pixel values to be between 0 and 1
# This is a key step from your PDF (Data Preprocessing)
train_images, test_images = train_images / 255.0, test_images / 255.0

print("Dataset loaded and normalized!")

# 3. Let's verify the data by looking at the first image
plt.imshow(train_images[0])
plt.show()

from tensorflow.keras import layers, models

# 3. Model Building: Create a simple CNN architecture
model = models.Sequential()
# First convolutional layer and pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
# Second convolutional layer and pooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# 4. Adding "Fully Connected" layers to the end
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) # 10 classes in CIFAR-10

print("CNN Model Architecture Created!")
model.summary()

# 5. Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Model compiled! Ready to start training...")

# 6. Train the model
# We use the training images and labels we loaded earlier
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# 7. Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(f'\nFinal Test Accuracy: {test_acc*100:.2f}%')


import numpy as np

# Define the labels for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 8. Make a prediction
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# 9. Show a test image and the AI's guess
img_index = 0 # You can change this number to see different images
plt.figure(figsize=(6,3))
plt.imshow(test_images[img_index])
predicted_label = np.argmax(predictions[img_index])
actual_label = test_labels[img_index][0]

plt.title(f"AI Guess: {class_names[predicted_label]} \nActual: {class_names[actual_label]}")
plt.show()

# Save the model to a file
model.save('my_cifar10_model.h5')
print("Model saved as my_cifar10_model.h5")