import cv2
from ultralytics import YOLO

# Liste des noms de classes COCO
nom_classes = ["personne", "velo", "voiture", "moto", "avion", "bus", "train", "camion",
               "bateau", "feu de signalisation", "borne d'incendie", "panneau d'arret", "parcmètre", "banc",
               "oiseau", "chat", "chien", "cheval", "mouton", "vache", "elephant", "ours", "zebre", "girafe",
               "sac à dos", "parapluie", "sac à main", "cravate", "valise", "frisbee", "skis", "snowboard",
               "ballon de sport", "cerf-volant", "batte de baseball", "gant de baseball", "planche à roulettes", "planche de surf",
               "raquette de tennis", "bouteille", "verre à vin", "tasse", "fourchette", "couteau", "cuillere", "bol", "banane",
               "pomme", "sandwich", "orange", "brocoli", "carotte", "hot-dog", "pizza", "beignet", "gâteau",
               "chaise", "canapé", "plante en pot", "lit", "table à manger", "toilette", "moniteur", "ordinateur portable", "souris",
               "telecommande", "clavier", "telephone portable", "micro-ondes", "four", "grille-pain", "evier", "refrigerateur",
               "livre", "horloge", "vase", "ciseaux", "ours en peluche", "sèche-cheveux", "brosse à dents"]

# Initialisation de la webcam
cap = cv2.VideoCapture(0)


# Chargement du modèle YOLOv8
modele = YOLO('yolov8l.pt')

while True:
    succes, img = cap.read()

    # Prédiction
    resultats = modele(img, stream=True)

    for resultat in resultats:
        boxes = resultat.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            classe_id = int(box.cls[0])

            classe = nom_classes[classe_id] if classe_id < len(nom_classes) else f"classe {classe_id}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f'{classe} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Affichage de la frame
    cv2.imshow('Détection YOLOv8', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
