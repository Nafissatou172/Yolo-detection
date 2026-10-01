# Real-Time Vision: Object Detection & Hand Tracking

## 🎯 Aperçu du Projet & Mission
Ce projet est une application de vision par ordinateur en temps réel conçue pour la détection d'objets et le suivi des mains (hand tracking) via une webcam.
**Mission :** Démontrer l'intégration de modèles d'Intelligence Artificielle de pointe (YOLOv8 et MediaPipe) pour créer une expérience visuelle interactive et réactive, avec une localisation des classes d'objets en français. Ce projet constitue une excellente vitrine de mes compétences en Computer Vision, en Python, et en manipulation de flux vidéo en temps réel.

## ✨ Fonctionnalités Principales
- **Détection d'Objets en Temps Réel :** Utilisation de YOLOv8 (versions *Nano* et *Large*) pour détecter, classifier et encadrer plus de 80 types d'objets différents simultanément.
- **Localisation (Français) :** Les noms de classes originaux du dataset COCO ont été traduits en français pour une expérience utilisateur localisée.
- **Suivi des Mains (Hand Tracking) :** Intégration de MediaPipe pour repérer et suivre avec précision les 21 points de repère (landmarks) de la main en temps réel.
- **Annotations Vidéo :** Dessin dynamique de boîtes de délimitation (bounding boxes), de probabilités de confiance, et des connexions de la main directement sur le flux vidéo.

## 🛠️ Technologies Utilisées
- **Langage :** Python 3
- **Computer Vision :** OpenCV (`opencv-python`)
- **Deep Learning / Object Detection :** Ultralytics YOLOv8
- **Hand Tracking :** Google MediaPipe
- **Traitement de données :** NumPy

## 📁 Structure du Projet
- `yolo_code.py` / `yolo_basic_detecte.py` : Scripts de base pour la détection d'objets avec YOLOv8.
- `yolo_flux_video_fr.py` : Script optimisé pour la détection en temps réel avec traduction des classes en français.
- `Main_LOXO.py` : Script dédié à la détection et au suivi des mouvements de la main avec MediaPipe.
- `indications.txt` : Instructions d'installation et liens utiles (ex: plateforme CVAT pour la labélisation).
- `*.pt` & `*.task` : Poids des modèles pré-entraînés (YOLOv8 et MediaPipe).

## 🚀 Installation et Démarrage

### 1. Cloner le dépôt
```bash
git clone <votre-url-github>
cd Yolo-detection
```

### 2. Créer et activer un environnement virtuel
```bash
python3 -m venv mon_env
source mon_env/bin/activate  # Sur macOS/Linux
# mon_env\Scripts\activate   # Sur Windows
```

### 3. Installer les dépendances
```bash
pip install opencv-python mediapipe ultralytics numpy
```

### 4. Exécuter les scripts
Pour lancer la détection d'objets en français :
```bash
python yolo_flux_video_fr.py
```
Pour lancer le suivi des mains :
```bash
python Main_LOXO.py
```
*(Appuyez sur la touche 'q' pour fermer la fenêtre vidéo)*

## 📈 Améliorations Futures
- Intégrer une interface utilisateur graphique (GUI) avec Tkinter ou PyQt.
- Combiner YOLOv8 et MediaPipe dans un seul et même flux vidéo pour interagir avec les objets détectés via des gestes.
- Entraîner un modèle YOLO personnalisé sur des données spécifiques labélisées via CVAT.
