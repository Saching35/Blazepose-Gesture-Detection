import cv2
import mediapipe as mp
import numpy as np



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
f = open("hand1.txt", "w")
f.write("Hand Gesture")

def calculateAngle(posx,posy,posz):
    a = np.array([posx.x, posx.y,posx.z])
    b = np.array([posy.x, posy.y, posy.z])
    c = np.array([posz.x, posz.y, posz.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
    
    
    
def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    color = (0, 0, 0)
    thickness = 1
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    #cv2.rectangle(img, pos, (end_x, end_y), bg_color, int(thickness))
    cv2.putText(img, text, (pos[0],pos[1]), font_face, scale, color, 1, cv2.LINE_AA)




def detectGesture(results,image):
    indexAngle = calculateAngle(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP])
    middleAngle = calculateAngle(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
    ringAngle = calculateAngle(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_MCP],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.RING_FINGER_PIP])
    pinkyAngle = calculateAngle(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_MCP],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.PINKY_PIP])
    thumbAngle1 = calculateAngle(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_MCP],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_IP])
    thumbAngle2 = calculateAngle(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_CMC],results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_MCP])
    f.write("\nindexAngle\t")
    f.write(str(indexAngle))
    f.write("\nmiddleAngle\t")
    f.write(str(middleAngle))
    f.write("\nringAngle\t")
    f.write(str(ringAngle))
    f.write("\npinkyAngle\t")
    f.write(str(pinkyAngle))
    f.write("\nthumbAngle1\t")
    f.write(str(thumbAngle1))
    f.write("\nthumbAngle2\t")
    f.write(str(thumbAngle2))
    pos = [400,250]
    if(indexAngle>160.0 and middleAngle>160.0 and ringAngle>160.0 and pinkyAngle>160.0):
        draw_label(image,"Hand",pos, "red")
    if(indexAngle<160.0 and middleAngle<160.0 and ringAngle<160.0 and pinkyAngle<160.0):
        draw_label(image,"Thumbs Up",pos, "red")





hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)
  f.write("in")
  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      f.write(str(hand_landmarks))
      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    detectGesture(results,image)
  cv2.imshow('MediaPipe Hands', image)
  f.write("out")
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
f.close()