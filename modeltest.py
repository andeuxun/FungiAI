import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Définir le répertoire des données
base_dir = "20Champis"  # Répertoire contenant les 20 dossiers de champignons
img_size = (200, 200)   # Dimension des images
batch_size = 32         # Taille du batch

# Prétraitement des données
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,         # Réduisez la rotation
    width_shift_range=0.1,     # Réduisez le décalage
    height_shift_range=0.1,
    shear_range=0.1,           # Réduisez le cisaillement
    zoom_range=0.1,            # Réduisez le zoom
    horizontal_flip=True,
    fill_mode='nearest'
)


train_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'         # Ensemble d'entraînement
)

val_data = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'       # Ensemble de validation
)

# Réduction du learning rate si la validation stagne
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',        # Suivre la perte sur validation
    factor=0.5,                # Réduction du learning rate par un facteur de 0.5
    patience=3,                # Réduction après 3 époques sans amélioration
    min_lr=1e-7                # Limite minimale
)

# Charger le modèle InceptionV3 pré-entraîné
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_size + (3,))

# Débloquer certaines couches pour le fine-tuning
base_model.trainable = True
# Ne fine-tune que les dernières 50 couches
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Ajouter des couches personnalisées pour la classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),  # Augmentez à 512 neurones
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),  # Ajoutez une couche supplémentaire
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])


# Compiler le modèle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraîner le modèle
epochs = 15  # Nombre d'époques
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[reduce_lr]      # Ajouter le callback ici
)

# Tracer les courbes d'entraînement et de validation
plt.figure(figsize=(12, 6))

# Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()