import cv2
import os

# data_file path
dir_path = 'hand_gestures/basic'# stop

if not os.path.exists( dir_path ):
    os.makedirs( dir_path )
cnt = 0

cap = cv2.VideoCapture(0)
cap.set( 3, 176 )
cap.set( 4, 100 )
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    file_name = os.path.join( dir_path, str(cnt) + '.jpg' )
    if not fps%3:
        cv2.imwrite( file_name, frame )
        print(len(os.listdir( dir_path )))
        cnt+=1

    fps += 1

    if len(os.listdir( dir_path )) == 37:
        break

    key = cv2.waitKey( 1 ) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()