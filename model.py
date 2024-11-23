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
import pathlib
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Chemin vers le dossier de données
data_dir = pathlib.Path("archive")

# Vérification si le dossier existe
if not data_dir.exists():
    raise FileNotFoundError(f"Le dossier {data_dir} n'existe pas. Vérifiez le chemin.")

# Sélectionner uniquement les 20 premiers sous-dossiers
subdirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])[:20]
selected_classes = [d.name for d in subdirs]
print("Sous-dossiers sélectionnés :", selected_classes)

# Définir les paramètres des images
batch_size = 32
img_height = 199  # Taille adaptée à InceptionV3
img_width = 199

# Créer les datasets pour l'entraînement et la validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    class_names=selected_classes  # Limiter aux 20 sous-dossiers
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    class_names=selected_classes  # Limiter aux 20 sous-dossiers
)

# Préparer les datasets pour de meilleures performances
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Prétraitement des données avec preprocess_input
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.3),
        layers.RandomTranslation(0.2, 0.2),
    ]
)

base_model = tf.keras.applications.InceptionV3(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Débloquer les dernières couches pour fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Garder les 50 dernières couches "entraînables"
    layer.trainable = False

# Définir le nombre de classes
num_classes = len(selected_classes)

# Construire le modèle
model = Sequential([
    data_augmentation,
    layers.Lambda(preprocess_input),  # Prétraitement spécifique au modèle
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile avec un optimiseur et un learning rate scheduler
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


# Callbacks pour EarlyStopping et ajustement du LR
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6, 
    verbose=1
)

# Entraînement du modèle
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stopping, reduce_lr]
)


# Callbacks pour EarlyStopping et ajustement du LR
callbacks = [early_stopping, reduce_lr]

# Phase 1 : Entraînement initial avec les couches gelées
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stopping, reduce_lr]  # Callbacks déjà définis
)

# Dégelez certaines couches pour fine-tuning
base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2  # Dégeler les dernières couches

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Learning rate plus faible pour fine-tuning
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Entraîner à nouveau avec fine-tuning
history_fine_tune = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)


# visualiser les résultats de l'entraînement
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = 20
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
# sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
# sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

# img = tf.keras.utils.load_img(
#    sunflower_path, target_size=(img_height, img_width)
# )

# img_array = tf.keras.utils.img_to_array(img) # Convertir en tableau numpy
# img_array = tf.expand_dims(img_array, 0) # Créer un batch

# predictions = model.predict(img_array) # Prédire sur le batch
# score = tf.nn.softmax(predictions[0]) # Appliquer une fonction softmax

# print(
#    "Cette image appartient très probablement à {} avec un taux de pourcentage de  {:.2f}."
#    .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
