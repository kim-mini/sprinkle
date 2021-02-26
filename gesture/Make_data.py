import cv2
import os

# data_file path
file_name = 'go_away'# stop
file_path = os.path.join( 'data/01', file_name )

# if not os.path.exists( file_path ):
#     os.mkdir( file_path )


cnt = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Video', frame)

    imageFile = os.path.join( file_path, str(cnt) + '.jpg' )
    if cnt%40:
        cv2.imwrite( imageFile, frame )
        cnt = '{}'.format( int(cnt) + 1 )


    if len(os.listdir( file_path )) == 37:
        break

    key = cv2.waitKey( 1 ) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()