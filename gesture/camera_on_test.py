import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('video', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.realese()
cv2.destroyAllWindow()
