import cv2
import mediapipe as mp
import pygame
import time

# Initialize sound
pygame.mixer.init()

chord_A = pygame.mixer.Sound("A.wav")
chord_C = pygame.mixer.Sound("C.wav")
chord_D = pygame.mixer.Sound("D.wav")
chord_G = pygame.mixer.Sound("G.wav")
chord_E = pygame.mixer.Sound("E.wav")

current_chord = chord_A
current_name = "A"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_y = None
last_play = 0

while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            label = handedness.classification[0].label
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark

            # LEFT HAND → CHORD SELECTION
            if label == "Left":

                fingers = 0

                if lm[8].y < lm[6].y:
                    fingers += 1
                if lm[12].y < lm[10].y:
                    fingers += 1
                if lm[16].y < lm[14].y:
                    fingers += 1
                if lm[20].y < lm[18].y:
                    fingers += 1

                if fingers == 0:
                    current_chord = chord_E
                    current_name = "E"

                elif fingers == 1:
                    current_chord = chord_A
                    current_name = "A"

                elif fingers == 2:
                    current_chord = chord_C
                    current_name = "C"

                elif fingers == 3:
                    current_chord = chord_D
                    current_name = "D"

                elif fingers >= 4:
                    current_chord = chord_G
                    current_name = "G"

            # RIGHT HAND → STRUM
            if label == "Right":

                y = int(lm[8].y * h)

                if prev_y is not None:
                    speed = abs(y - prev_y)

                    if speed > 50 and time.time() - last_play > 0.25:
                        current_chord.play()
                        last_play = time.time()

                prev_y = y

    # UI
    cv2.putText(img,"AIR GUITAR AI",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(img,f"Chord: {current_name}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.putText(img,"Left hand = chord",(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(img,"Right hand = strum",(20,150),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(img,"0=E 1=A 2=C 3=D 4=G",(20,180),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(img,"Try Pattern: E  A  D  G",(20,210),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.imshow("Air Guitar",img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()