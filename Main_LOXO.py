import cv2
import mediapipe as mp
import numpy as np

loxo_modele = mp.solutions.hands
capture_loxo = loxo_modele.Hands()
mp_dessin = mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resultats = capture_loxo.process(imgRGB)
    if resultats.multi_hand_landmarks:
            for point_repere in resultats.multi_hand_landmarks:
                print(point_repere)
                mp_dessin.draw_landmarks(img, point_repere)
                for id , positions in enumerate(point_repere.landmark): # on charge les coordonnées (position) des landmark et leur id
                    h,w,c=img.shape      #les coordonnées x et y c'est des valeurs normalisées avec les dimension de l'image
                    cx,cy=int(positions.x*w) , int(positions.y *h)  #on les converti en pixel
                    print(id,cx,cy)
                    if id == 8:        #mettre en exergue le landmark à la position 0, ou 4 ou Tous (en enlevant le if)
                        cv2.circle(img, (cx,cy), 15, (0,255,0), cv2.FILLED)
                mp_dessin.draw_landmarks(img, point_repere,loxo_modele.HAND_CONNECTIONS)



    #imgRGB = np.flip(imgRGB,axis=1)
    cv2.imshow('ma fig', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()