import cv2
import numpy as np

def kamera_ac():
    kamera = cv2.VideoCapture(0)
    if not kamera.isOpened():
        print("Kamera açılamadı!")
        exit()
    return kamera

def maske_olustur(frame, alt, ust):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, alt, ust)
    kernel = np.ones((5, 5), np.uint8)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, kernel)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel)
    return maske

def kontur_ciz(frame, maske):
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for kontur in konturlar:
        alan = cv2.contourArea(kontur)
        if alan > 1000:
            x, y, w, h = cv2.boundingRect(kontur)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return frame

def main():
    alt_mavi = np.array([90, 50, 50])
    ust_mavi = np.array([130, 255, 255])
    kamera = kamera_ac()

    while True:
        ret, frame = kamera.read()
        if not ret:
            print("Kare alınamadı.")
            break

        maske = maske_olustur(frame, alt_mavi, ust_mavi)
        frame = kontur_ciz(frame, maske)

        cv2.imshow("Kamera", frame)
        cv2.imshow("Maske", maske)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kamera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()