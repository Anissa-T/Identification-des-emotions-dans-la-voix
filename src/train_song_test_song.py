import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactive l'utilisation du GPU

import random
import librosa
import numpy as np
from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paramètres constants
SAMPLE_RATE = 22050
MFCC_FEATURES = 40
MAX_LEN = 300
DATASET_PATH = "../data/song/"
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def set_seed(seed: int = 42):
    """Fixer les graines pour la reproductibilité."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Extraction des caractéristiques MFCC
def extract_mfcc(file_path: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Extraction des caractéristiques MFCC d'un fichier audio."""
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

# Chargement du dataset
def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Charger le dataset et retourner les features et labels."""
    features, labels = [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_code = file.split("-")[2]
                if emotion_code in EMOTION_MAP:
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(EMOTION_MAP[emotion_code])
    return np.array(features), np.array(labels)

def preprocess_labels(labels: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    """Encodage des labels texte en format numérique."""
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder

def get_lr_scheduler():
    """Réduction du learning rate si la validation stagne."""
    return ReduceLROnPlateau(
        monitor='val_loss',   # Surveiller la perte de validation
        factor=0.5,           # Réduction de moitié du learning rate
        patience=5,           # Nombre d'epochs sans amélioration avant réduction
        min_lr=1e-6,          # Learning rate minimal
        verbose=1             # Afficher les changements
    )

# Création du modèle CNN
def create_cnn_model(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """Créer et compiler le modèle CNN avec régularisation L2 et Dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        # Première couche Conv1D avec régularisation L2
        tf.keras.layers.Conv1D(
            64, kernel_size=3, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)  # Ajout de la régularisation L2
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Deuxième couche Conv1D avec régularisation L2
        tf.keras.layers.Conv1D(
            128, kernel_size=3, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),

        # Flatten et couches denses avec Dropout
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128, activation='relu',
            kernel_regularizer=regularizers.l2(0.001)  # Régularisation L2 pour la couche Dense
        ),
        tf.keras.layers.Dropout(0.5),  # Augmentation du taux de Dropout à 50%

        # Couche de sortie
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Fonction pour tracer les métriques de performance
def plot_training_history(history: tf.keras.callbacks.History):
    """Tracer les courbes de perte et précision pour l'entraînement et la validation."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))

    # Précision
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Précision Entraînement')
    plt.plot(epochs, val_acc, 'r', label='Précision Validation')
    plt.title('Précision Entraînement et Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Précision')
    plt.legend()

    # Pertes
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Perte Entraînement')
    plt.plot(epochs, val_loss, 'r', label='Perte Validation')
    plt.title('Perte Entraînement et Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../plots/train_song_test_song_-_training_validation_metrics.png')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN Confusion Matrix')
    plt.xlabel('Labels Prédits')
    plt.ylabel('Labels Réels')
    plt.savefig('../plots/train_song_test_song_-_confusion_matrix.png.png')
    plt.show()

# Main
def main():
    """Pipeline principal pour l'entraînement et évaluation du modèle CNN."""
    set_seed(42)

    print("Chargement du dataset...")
    features, labels = load_dataset(DATASET_PATH)

    print("Prétraitement des labels...")
    labels, encoder = preprocess_labels(labels)

    print("Normalisation des caractéristiques...")
    features = features / np.max(features)  # Normalisation des données

    print("Division du dataset...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print("Création du modèle CNN...")
    input_shape = (MAX_LEN, MFCC_FEATURES)
    num_classes = len(np.unique(labels))
    cnn_model = create_cnn_model(input_shape, num_classes)

    print("Entraînement du modèle CNN...")
    lr_scheduler = get_lr_scheduler()
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[lr_scheduler])

    train_loss, train_acc = cnn_model.evaluate(X_train, y_train)
    print(f"##### Accuracy Train: {train_acc:.4f} #####")
    print(f"##### Loss Train: {train_loss:.4f} #####")

    print("Évaluation sur le jeu de test...")
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
    print(f"##### Accuracy Test: {test_acc:.4f} #####")
    print(f"##### Loss Test: {test_loss:.4f} #####")

    # Tracer les métriques d'entraînement
    plot_training_history(history)

    # Pour la matrice de confusion : prédit les labels pour le set de test
    y_pred = cnn_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot confusion matrix
    #class_names = LabelEncoder().fit(labels).classes_  # Obtenir les noms des classes depuis label encoder
    class_names = encoder.classes_
    plot_confusion_matrix(y_test, y_pred_classes, class_names)

if __name__ == "__main__":
    main()
