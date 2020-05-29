print("\t---------Initialize hyperparameters---------\n")
steps = int(input("Enter steps_per_epoch: "))
no_of_epoch = int(input("Enter no_of_epoch: "))
    

# Import important libraries

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model

from keras.preprocessing import image

import numpy as np

# Create the model

model = Sequential()

# Add input layer

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))

# Add pooling layer

model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding middle layers and output layer

model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       )) # Relu activation function

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

# Compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=steps,
        epochs=no_of_epoch,
        validation_data=test_set,
        validation_steps=800)

# Load pre-trained model

m = load_model('cnn-cat-dog-model.h5')

# Load image to test result

test_image = image.load_img('cnn_dataset/test_set/cats/cat.4003.jpg', 
               target_size=(64,64))

# Convert test image to np array

test_image = image.img_to_array(test_image)

# Change dimnsions of array

test_image = np.expand_dims(test_image, axis=0)

# Predict the result

result = m.predict(test_image)

# Display the result

if result[0][0] == 1.0:
    print('dog')
else:
    print('cat')
    
accuracy = int(model.history.history['val_accuracy'][-1]*100)

print("final accuracy ",accuracy)