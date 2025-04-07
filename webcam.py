import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("C:/Users/murat/Desktop/duygu_analizi/best_model_2.h5")

# Duygu etiketleri
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Webcam'i başlatmak için
cap = cv2.VideoCapture(0)

# Yüz algılamak
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüz algılama
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    for (x, y, w, h) in faces:
        # Yüz bölgesini al
        face = gray_frame[y:y+h, x:x+w]
        
        # Yüzü model giriş boyutuna göre yeniden boyutlandır
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0  
        face_input = np.expand_dims(face_normalized, axis=(0, -1))

        # Model ile tahmin yap
        prediction = model.predict(face_input)
        emotion_label = np.argmax(prediction)
        emotion = emotions[emotion_label]

        # Çerçeve çiz ve tahmini yaz
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    cv2.imshow("Emotion Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
