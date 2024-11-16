"""
    Ne pas oublier de travailler dans un environnement virtuel : 
        >> python3 -m venv tensorflow_env
        >> source tensorflow_env/bin/activate
        >> pip install tensorflow
    
"""

# Désactiver les messages d'erreur de TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
import PIL # pip install pillow
import pathlib
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Sequential


data_dir = "/home/thomas/.keras/datasets/champi"
data_dir = pathlib.Path(data_dir)


# un répertoire d'image sur disque à un tf.data.Dataset

batch_size = 32 #  d'actualiser les paramètres du modèle après chaque batch au lieu de le faire après chaque échantillon ou seulement après avoir vu l'ensemble complet
img_height = 120
img_width = 120

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
"""
(32, 180, 180, 3) --> 32 images de taille 180x180 pixels et 3 canaux (RGB)
(32,) --> ce sont des labels correspondants aux 32 images
"""


AUTOTUNE = tf.data.AUTOTUNE
# améliorer les performances de lecture des données
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) #shuffle est utilisé pour mélanger les données 1000 veut dire que le modèle va mélanger les données par paquets de 1000
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# Créer le modèle séquentiel

num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # désactiver 20% des neurones
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# Compilation du modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Résumé du modèle
model.summary()


epochs=15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



# visualiser les résultats de l'entraînement
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Prédire sur de nouvelles données
#sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
#sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

#img = tf.keras.utils.load_img(
#    sunflower_path, target_size=(img_height, img_width)
#)

#img_array = tf.keras.utils.img_to_array(img) # Convertir en tableau numpy
#img_array = tf.expand_dims(img_array, 0) # Créer un batch

#predictions = model.predict(img_array) # Prédire sur le batch
#score = tf.nn.softmax(predictions[0]) # Appliquer une fonction softmax

#print(
#    "Cette image appartient très probablement à {} avec un taux de pourcentage de  {:.2f}."
#    .format(class_names[np.argmax(score)], 100 * np.max(score))
#)
