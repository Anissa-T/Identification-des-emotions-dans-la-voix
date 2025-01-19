# Importation des bibliothèques nécessaires
import os  # Gestion des opérations système (par exemple, chemins de fichiers)
import random  # Génération de nombres aléatoires
import librosa  # Traitement des fichiers audio (par exemple, chargement et extraction de caractéristiques)
import numpy as np  # Calcul scientifique et manipulation de tableaux
import tensorflow as tf  # Framework d'apprentissage profond
from tensorflow.keras import regularizers  # Régularisation L2 pour éviter le surapprentissage
from tensorflow.keras.callbacks import ReduceLROnPlateau  # Réduction dynamique du taux d'apprentissage
from sklearn.model_selection import train_test_split  # Division des données en ensembles d'entraînement/test
from sklearn.preprocessing import LabelEncoder  # Encodage des labels textuels en entiers
from sklearn.metrics import confusion_matrix  # Calcul de la matrice de confusion
import matplotlib.pyplot as plt  # Visualisation des données (graphiques, courbes)
import seaborn as sns  # Visualisation avancée (par exemple, heatmaps)
import parselmouth  # Analyse acoustique (par exemple, mesure de la fréquence fondamentale)
from parselmouth.praat import call  # Utilisation de fonctions Praat via Parselmouth
from typing import List, Tuple  # Annotation des types pour les fonctions

# Désactivation de l'utilisation du GPU (utile pour les environnements CPU uniquement)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Définition des paramètres constants
SAMPLE_RATE = 22050  # Fréquence d'échantillonnage pour le traitement audio
MFCC_FEATURES = 40  # Nombre de coefficients MFCC extraits
MAX_LEN = 300  # Longueur maximale des caractéristiques MFCC après padding
DATASET_PATH = "../data/speech/"  # Chemin vers le dossier contenant les fichiers audio
EMOTION_MAP = {  # Mappage des codes d'émotion aux étiquettes textuelles
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
    - Charge le fichier audio avec une fréquence d'échantillonnage fixe.
    - Extrait les coefficients MFCC.
    - Applique un padding ou tronque les caractéristiques pour atteindre une longueur uniforme.
    """
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)  # Chargement du fichier audio
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=MFCC_FEATURES)  # Extraction des MFCC
    if mfcc.shape[1] < max_len:  # Si les MFCC sont plus courts que la longueur maximale
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')  # Padding
    else:
        mfcc = mfcc[:, :max_len]  # Troncature si trop long
    return mfcc.T  # Transposition pour que les dimensions correspondent au modèle

def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Charger les fichiers audio et leurs étiquettes à partir du dataset.
    - Parcourt récursivement les fichiers dans le chemin spécifié.
    - Charge les fichiers .wav, extrait les MFCC et assigne les étiquettes correspondantes.
    """
    features, labels, file_paths = [], [], []
    for root, _, files in os.walk(dataset_path):  # Parcours récursif des dossiers
        for file in files:
            if file.endswith(".wav"):  # Seuls les fichiers audio au format WAV sont pris en compte
                file_path = os.path.join(root, file)
                emotion_code = file.split("-")[2]  # Extraction du code d'émotion depuis le nom du fichier
                if emotion_code in EMOTION_MAP:  # Vérification que le code est valide
                    mfcc = extract_mfcc(file_path)  # Extraction des MFCC
                    features.append(mfcc)
                    labels.append(EMOTION_MAP[emotion_code])  # Ajout de l'étiquette correspondante
                    file_paths.append(file_path)
    return np.array(features), np.array(labels), file_paths

def preprocess_labels(labels: np.ndarray) -> Tuple[np.ndarray, LabelEncoder]:
    """Convertir les étiquettes textuelles en valeurs numériques à l'aide de LabelEncoder."""
    encoder = LabelEncoder()
    return encoder.fit_transform(labels), encoder

def get_lr_scheduler():
    """Créer un callback pour réduire dynamiquement le taux d'apprentissage si la validation stagne."""
    return ReduceLROnPlateau(
        monitor='val_loss',  # Surveiller la perte de validation
        factor=0.5,  # Diviser le taux d'apprentissage par 2 en cas de stagnation
        patience=5,  # Nombre d'époques sans amélioration avant de réduire
        min_lr=1e-6,  # Limite inférieure pour le taux d'apprentissage
        verbose=1  # Afficher les messages dans la console
    )

def create_cnn_model(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    """
    Construire un modèle CNN pour la classification des émotions audio.
    - Inclut une régularisation L2 pour limiter le surapprentissage.
    - Utilise Dropout pour améliorer la généralisation.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # Entrée du modèle
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Première couche convolutionnelle
        tf.keras.layers.MaxPooling1D(pool_size=2),  # Pooling pour réduire les dimensions
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Deuxième couche convolutionnelle
        tf.keras.layers.MaxPooling1D(pool_size=2),  # Deuxième couche de pooling
        tf.keras.layers.Flatten(),  # Conversion en vecteur 1D
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # Couche dense avec régularisation
        tf.keras.layers.Dropout(0.5),  # Dropout pour éviter le surapprentissage
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Couche de sortie pour classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compilation du modèle
    return model

def measure_f0(audio_file_path: str) -> float:
    """
    Mesurer la fréquence fondamentale (f0) à l'aide de Parselmouth.
    - Ignore les silences en excluant les fréquences nulles.
    """
    sound = parselmouth.Sound(audio_file_path)
    pitch = call(sound, "To Pitch", 0.0, 75, 600)  # Calcul du pitch
    pitch_values = pitch.selected_array['frequency']  # Extraction des valeurs de fréquence
    pitch_values = pitch_values[pitch_values != 0]  # Exclure les silences
    return np.mean(pitch_values) if len(pitch_values) > 0 else 0  # Moyenne ou 0 si aucun pitch trouvé

def plot_training_history(history: tf.keras.callbacks.History):
    """Tracer l'évolution de la précision et de la perte pour l'entraînement et la validation."""
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
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """Tracer la matrice de confusion pour visualiser les performances du modèle."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de confusion CNN')
    plt.xlabel('Labels Prédits')
    plt.ylabel('Labels Réels')
    plt.show()

def main():
    """Fonction principale pour exécuter les étapes de traitement et de classification."""
    set_seed(42)  # Fixer les graines pour la reproductibilité
    print("Chargement du dataset...")
    features, labels, file_paths = load_dataset(DATASET_PATH)  # Chargement des données
    print(f"Dataset chargé : {len(features)} fichiers audio traités.")

    print("Prétraitement des labels...")
    labels_encoded, encoder = preprocess_labels(labels)  # Encodage des labels
    print(f"Labels encodés : {len(set(labels_encoded))} classes détectées.")

    print("Normalisation des caractéristiques...")
    features = features / np.max(features)  # Normalisation des MFCC
    features = features[..., np.newaxis]  # Ajout d'une dimension pour compatibilité avec CNN
    print("Normalisation terminée.")

    print("Division du dataset...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)  # Division des données
    print(f"Ensemble d'entraînement : {len(X_train)} échantillons.")
    print(f"Ensemble de test : {len(X_test)} échantillons.")

    print("Création du modèle CNN...")
    input_shape = (MAX_LEN, MFCC_FEATURES)  # Dimensions des données d'entrée
    num_classes = len(np.unique(labels_encoded))  # Nombre de classes à prédire
    cnn_model = create_cnn_model(input_shape, num_classes)  # Création du modèle CNN
    print("Modèle CNN créé.")

    print("Début de l'entraînement du modèle...")
    lr_scheduler = get_lr_scheduler()  # Callback pour ajuster le taux d'apprentissage
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[lr_scheduler])  # Entraînement
    print("Entraînement terminé.")

    print("Évaluation sur le jeu de test...")
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)  # Évaluation des performances
    print(f"Résultats de l'évaluation :")
    print(f"  - Précision sur le test : {test_acc:.4f}")
    print(f"  - Perte sur le test : {test_loss:.4f}")

    # Analyse des fréquences fondamentales pour chaque émotion
    print("Analyse des fréquences fondamentales (f0) pour chaque émotion...")
    f0_data = {}
    for file, label in zip(file_paths, labels_encoded):
        emotion = encoder.inverse_transform([label])[0]  # Obtenir l'émotion en clair
        f0_value = measure_f0(file)  # Mesurer la fréquence fondamentale
        if emotion in f0_data:
            f0_data[emotion].append(f0_value)
        else:
            f0_data[emotion] = [f0_value]

    # Calcul des moyennes des fréquences fondamentales
    for emotion, f0_list in f0_data.items():
        average_f0 = np.mean(f0_list) if f0_list else 0
        print(f"  - Fréquence fondamentale moyenne pour {emotion}: {average_f0:.2f}")

    # Tracer les courbes d'entraînement et validation
    print("Génération des courbes d'entraînement et validation...")
    plot_training_history(history)
    print("Courbes générées.")

    # Génération de la matrice de confusion
    print("Prédictions sur le jeu de test...")
    y_pred = cnn_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("Génération de la matrice de confusion...")
    class_names = encoder.classes_
    plot_confusion_matrix(y_test, y_pred_classes, class_names)
    print("Matrice de confusion générée.")

    print("Processus terminé avec succès.")
    
if __name__ == "__main__":
    main()