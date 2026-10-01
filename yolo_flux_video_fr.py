import cv2
from ultralytics import YOLO

# Liste des noms de classes COCO traduits en français
noms_classes_fr = [
    "personne", "vélo", "voiture", "moto", "avion", "bus", "train", "camion",
    "bateau", "feu de signalisation", "borne d'incendie", "panneau d'arrêt", "parcmètre", "banc",
    "oiseau", "chat", "chien", "cheval", "mouton", "vache", "éléphant", "ours", "zèbre", "girafe",
    "sac à dos", "parapluie", "sac à main", "cravate", "valise", "frisbee", "skis", "snowboard",
    "ballon de sport", "cerf-volant", "batte de baseball", "gant de baseball", "skateboard", "planche de surf",
    "raquette de tennis", "bouteille", "verre à vin", "tasse", "fourchette", "couteau", "cuillère", "bol", "banane",
    "pomme", "sandwich", "orange", "brocoli", "carotte", "hot-dog", "pizza", "beignet", "gâteau",
    "chaise", "canapé", "plante en pot", "lit", "table à manger", "toilettes", "téléviseur", "ordinateur portable", "souris",
    "télécommande", "clavier", "téléphone portable", "micro-ondes", "four", "grille-pain", "évier", "réfrigérateur",
    "livre", "horloge", "vase", "ciseaux", "ours en peluche", "sèche-cheveux", "brosse à dents"
]

# Charger le modèle YOLOv8
modele = YOLO("yolov8l.pt")

# Remplacer les noms des classes par les traductions françaises
modele.model.names = {i: nom for i, nom in enumerate(noms_classes_fr)}

# Ouvrir le flux vidéo (0 = webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    exit()

# Définir la taille de la fenêtre vidéo
cap.set(3, 640)
cap.set(4, 480)

print("Appuyez sur 'q' pour quitter.")

while True:
    success, frame = cap.read()
    if not success:
        print("Erreur de lecture du flux vidéo.")
        break

    # Détection avec seuil de confiance minimum 0.5
    resultats = modele(frame, stream=True, conf=0.5)

    for resultat in resultats:
        image_annotée = resultat.plot()
        cv2.imshow("Détection en temps réel - YOLOv8 (FR)", image_annotée)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
