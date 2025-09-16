import cv2

url = "rtsp://admin:abcd1234@192.168.1.107:554/cam/realmonitor?channel=1?subtype=0"

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    exit()

while True:
    ret , frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()