import cv2
import mediapipe as mp

# Inicializa MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Inicializa MediaPipe Drawing Utilities.
mp_drawing = mp.solutions.drawing_utils

# Abre la cámara web.
cap = cv2.VideoCapture(1)#la 1 la puse para que jale la camara de droidcam, la 0 es la webcam

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convertir la imagen a RGB.
    #image = cv2.cvtColor(cv2.flip(image, 90), cv2.COLOR_BGR2RGB)

    # Procesar la imagen y obtener las poses detectadas.
    results = pose.process(image)

    # Dibuja las poses detectadas en la imagen.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Mostrar la imagen con las poses detectadas.
    cv2.imshow('Valute', image)

    # Salir si se presiona la tecla 'q'.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Limpieza y cierre de recursos.
cap.release()
cv2.destroyAllWindows()
