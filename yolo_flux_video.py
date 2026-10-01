import cv2
from ultralytics import YOLO

# Charger le modèle
modele = YOLO("yolov8l.pt")

# Ouvrir la webcam (ou remplace 0 par le chemin d'une vidéo locale)
cap = cv2.VideoCapture(0)  # ou "video.mp4"



while True:
    success, frame = cap.read()
    if not success:
        print("Fin du flux ou erreur de lecture.")
        break

    # Détection avec stream=True pour plus de contrôle
    resultats = modele(frame, stream=True, conf=0.5)

    for resultat in resultats:
        frame_annotee = resultat.plot()  # applique les boîtes sur l'image
        cv2.imshow("Flux vidéo - YOLOv8", frame_annotee)

    # Quitter avec la touche Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()

