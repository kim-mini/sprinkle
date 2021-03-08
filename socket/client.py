# -*- coding: utf8 -*-
import cv2
import socket
import numpy as np
import threading
from queue import Queue

HOST = 'awsbit.mynetgear.com'
PORT = 65223

## TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
## server ip, port
s.connect((HOST, PORT))


def gettingMsg():
    while True:
        data = s.recv(1024)
        if not data: break
        else:
           data = str(data).split("b'", 1)[1].rsplit("'",1)[0]
           print(data)
           if data == 'q':
              cam.release()
              s.close()

def sendVideo(VideoFrame):
    while True:
        frame = videoFrame.get()
        if videoFrame == None:
            break
        print('1')
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        # frame을 String 형태로 변환
        data = np.array(frame)
        #stringData = data.tostring()
        stringData = data.tobytes()
	 
        #서버에 데이터 전송
	    #(str(len(stringData))).encode().ljust(16)
        s.sendall((str(len(stringData))).encode().ljust(16) + stringData)
    s.close()
	 
if __name__ == '__main__':

    videoFrame = Queue()

    th1 = threading.Thread(target=gettingMsg, args=())
    th2 = threading.Thread(target=sendVideo, args=(videoFrame, ))
    th1.start()
    th2.start()

	 
	 
    ## webcam 이미지 capture
    cam = cv2.VideoCapture(0)
	 
    ## 이미지 속성 변경 3 = width, 4 = height
    #cam.set(3, 320)
    #cam.set(4, 240)
	 
    ## 0~100에서 90의 이미지 품질로 설정 (default = 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	 
    while True:
        # 비디오의 한 프레임씩 읽는다.
        # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
        ret, frame = cam.read()
        # cv2. imencode(ext, img [, params])
        # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
        if not ret:
            break
        videoFrame.put(frame)
        print('2')
    th1.join()
    th2.join()
    cam.release()
