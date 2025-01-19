# Importation des bibliothèques nécessaires
import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import parselmouth
from parselmouth.praat import call
from typing import List, Tuple

# Désactiver l'utilisation du GPU pour l'entraînement (utile si GPU indisponible ou pour des tests)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Définition des paramètres constants du projet
SAMPLE_RATE = 22050  # Taux d'échantillonnage pour les fichiers audio
MFCC_FEATURES = 40   # Nombre de coefficients MFCC à extraire
MAX_LEN = 300        # Longueur maximale des vecteurs MFCC après padding
DATASET_PATH = "../data/song/"  # Chemin vers le dataset audio
# Mappage des codes d'émotion (dans les noms de fichiers) aux étiquettes textuelles
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
    """Fixer les graines pour garantir la reproductibilité des expériences."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def extract_mfcc(file_path: str, max_len: int = MAX_LEN) -> np.ndarray:
    """
    Extraire les caractéristiques MFCC d'un fichier audio.
    - Charge le fichier audio avec un taux d'échantillonnage fixe.
    - Extrait les MFCC.
    - Ajoute du padding ou tronque les données pour obtenir une longueur constante.
    """
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Charger les fichiers audio du dataset et extraire leurs MFCC et étiquettes.
    - Parcourt récursivement les sous-dossiers.
    - Ne traite que les fichiers terminant par ".wav".
    - Associe chaque fichier à une étiquette d'émotion en utilisant EMOTION_MAP.
    """
    features, labels, file_paths, emotions = [], [], [], []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion_code = file.split("-")[2]  # Extraire le code d'émotion du nom du fichier
                if emotion_code in EMOTION_MAP:
                    mfcc = extract_mfcc(file_path)
                    features.append(mfcc)
                    labels.append(EMOTION_MAP[emotion_code])
                    file_paths.append(file_path)
                    emotions.append(EMOTION_MAP[emotion_code])
    return np.array(features), np.array(labels), file_paths, emotions

def preprocess_labels(labels: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    """Encoder les labels textuels en entiers pour le modèle."""
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    return labels_encoded, encoder

def get_lr_scheduler():
    """Créer un callback pour réduire le taux d'apprentissage si la validation stagne."""
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

def create_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """
    Définir et compiler un modèle CNN pour la classification.
    - Inclut la régularisation L2 pour éviter le surapprentissage.
    - Utilise Dropout pour renforcer la généralisation.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def measure_f0(audio_file_path):
    """
    Mesurer la fréquence fondamentale (f0) d'un fichier audio avec Parselmouth.
    - Ignore les silences en ne considérant que les fréquences non nulles.
    """
    sound = parselmouth.Sound(audio_file_path)
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0]  # Exclure les silences
    return np.mean(pitch_values) if len(pitch_values) > 0 else 0

def plot_training_history(history: tf.keras.callbacks.History):
    """Tracer les courbes de précision et de perte pour l'entraînement et la validation."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Précision Entraînement')
    plt.plot(epochs, val_acc, 'r', label='Précision Validation')
    plt.title('Précision Entraînement et Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Précision')
    plt.legend()

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
    """Tracer la matrice de confusion basée sur les étiquettes réelles et prédites."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion CNN')
    plt.xlabel('Labels Prédits')
    plt.ylabel('Labels Réels')
    plt.savefig('../plots/train_song_test_song_-_confusion_matrix.png')
    plt.show()

def main():
    """Point d'entrée principal pour charger les données, entraîner le modèle et évaluer les performances."""
    print("Initialisation du processus...")
    set_seed(42)  # Fixer les graines pour la reproductibilité
    print("Graines initialisées.")

    # Chargement des données
    print("Chargement du dataset...")
    features, labels, file_paths, emotions = load_dataset(DATASET_PATH)
    print(f"Dataset chargé : {len(features)} fichiers audio traités.")

    # Prétraitement des labels
    print("Prétraitement des labels...")
    labels_encoded, encoder = preprocess_labels(labels)
    print(f"Labels encodés : {len(set(labels_encoded))} classes détectées.")

    # Normalisation des caractéristiques
    print("Normalisation des caractéristiques...")
    features = features / np.max(features)  # Échelle des caractéristiques entre 0 et 1
    features = features[..., np.newaxis]  # Ajouter une dimension de canal pour le CNN
    print("Normalisation terminée.")

    # Division des données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test, train_files, test_files, train_emotions, test_emotions = train_test_split(
        features, labels_encoded, file_paths, emotions, test_size=0.2, random_state=42)
    print(f"Ensemble d'entraînement : {len(X_train)} échantillons.")
    print(f"Ensemble de test : {len(X_test)} échantillons.")

    # Création et entraînement du modèle CNN
    print("Création du modèle CNN...")
    input_shape = (MAX_LEN, MFCC_FEATURES, 1)
    num_classes = len(np.unique(labels_encoded))
    cnn_model = create_cnn_model(input_shape, num_classes)
    print("Modèle CNN créé.")

    print("Configuration du callback pour ajuster le taux d'apprentissage...")
    lr_scheduler = get_lr_scheduler()

    print("Début de l'entraînement du modèle...")
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[lr_scheduler])
    print("Entraînement terminé.")

    # Évaluation du modèle
    print("Évaluation du modèle sur le jeu de test...")
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
    print(f"Résultats de l'évaluation :")
    print(f"  - Précision sur le test : {test_acc:.4f}")
    print(f"  - Perte sur le test : {test_loss:.4f}")

    # Analyse des fréquences fondamentales pour chaque émotion
    print("Analyse des fréquences fondamentales (f0) pour chaque émotion...")
    f0_data = {}
    for emotion in set(test_emotions):
        print(f"Calcul des fréquences fondamentales pour l'émotion : {emotion}")
        emotion_files = [file for file, e in zip(test_files, test_emotions) if e == emotion]
        f0_measurements = [measure_f0(file) for file in emotion_files]
        f0_data[emotion] = np.mean(f0_measurements)
        print(f"  - Fréquence fondamentale moyenne pour {emotion} : {f0_data[emotion]:.2f}")

    # Tracer les métriques d'entraînement et la matrice de confusion
    print("Génération des courbes d'entraînement et validation...")
    plot_training_history(history)
    print("Courbes générées.")

    print("Prédictions sur le jeu de test...")
    y_pred = cnn_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Génération de la matrice de confusion...")
    class_names = encoder.classes_
    plot_confusion_matrix(y_test, y_pred_classes, class_names)
    print("Matrice de confusion générée.")

    print("Processus terminé avec succès.")
    
# Exécution du script principal
if __name__ == "__main__":
    main()