#pip install ultralytics
from ultralytics import YOLO
import cv2

#model1 =YOLO('yolov8l.pt') #n = nano (3151904 parameters)  l =large (43668288 parameters) et m = medium
model2 = YOLO('yolov8n.pt') # On peut  voir les fichiers des parametres (weigths)
#on peut les enrégistrer dans un repertoire yolo_poids
#model3=YOLO('poids_yolo/yolov8l.pt')


resultat = model2("1.jpg",show=True)

cv2.waitKey(0)