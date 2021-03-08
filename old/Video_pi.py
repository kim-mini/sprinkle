import tensorflow as tf
import cv2
import time
import numpy as np
import socket
from PIL import ImageDraw, ImageFont, Image

# tcp로 연결한 라즈베리파이 연결
HOST='192.168.1.4'
PORT=65223

#socket에서 수신한 버퍼를 반환하는 함수
def recvall( sock, count ):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv( count )
        if not newbuf: return None
        buf += newbuf
        count -= len( newbuf )
    return buf

#TCP 사용
s=socket.socket( socket.AF_INET,socket.SOCK_STREAM )
print( 'Socket created' )

#서버의 아이피와 포트번호 지정
s.bind(( HOST, PORT ))
print( 'Socket bind complete' )
# 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개>까지 받는다)
s.listen( 10 )
print( 'Socket now listening' )

#연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
conn,addr=s.accept()

##########모델 로드

labels = [ 'go_away' ]

model = tf.keras.models.load_model('models/hand_gestures_classification_model/saved_model')

##########모델 예측

images = []
label = 0
confidence = 0
while True:
    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    data = np.fromstring( stringData, dtype = 'uint8' )
    #data를 디코딩한다.
    image = cv2.imdecode( data, cv2.IMREAD_COLOR )

    #print(type(image)) #<class 'numpy.ndarray'>
    #print(image.shape) #(720, 1280, 3)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    #print(rgb_image.shape) #(224, 224)
    # Our sequence length (36) of images in one video is huge and fps of camera is quite low.
    # To fill the buffer quickly, fill three images.
    images.append(rgb_image)
    images.append(rgb_image)
    images.append(rgb_image)

    if len(images) == 30:
        x_test = np.array([images])
        x_test = x_test / 255

        y_predict = model.predict(x_test)

        label = labels[y_predict[0].argmax()]
        confidence = y_predict[0][y_predict[0].argmax()]
    
        images.clear()
    cv2.putText(image, text='{} {:.2f}%'.format(label, confidence * 100), org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


    cv2.imshow('image', image)

    if cv2.waitKey(1) == ord('q'): #key 입력이 있을때까지 1밀리 세컨드 만큼 대기
        break

cv2.destroyAllWindows()
