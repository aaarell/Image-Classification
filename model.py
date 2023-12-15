import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

# Path to your dataset directory
train_dir = r"C:\Users\Maudyy\PycharmProjects\Prak Modul 6\rps\train"
val_dir = r"C:\Users\Maudyy\PycharmProjects\Prak Modul 6\rps\val"
test_dir = r"C:\Users\Maudyy\PycharmProjects\Prak Modul 6\rps\test"

# Image size
img_size = (224, 224)

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load Dataset
train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                                    batch_size=32, class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(val_dir, target_size=img_size,
                                                              batch_size=32, class_mode='categorical')

# Create MobileNet base model
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size,
                                                  batch_size=32, class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save("mobilenet_model.h5")