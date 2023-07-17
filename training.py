import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# path directory
train_dir = r'C:\Users\C\Desktop\ClassifAI\train'
valid_dir = r'C:\Users\C\Desktop\ClassifAI\validation'

# scale img
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['cats', 'dogs']
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    classes=['cats', 'dogs']
)

model = tf.keras.models.Sequential([
    tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compute
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // batch_size
)

# save model
model.save('ClassifAI/trained_model.h5')