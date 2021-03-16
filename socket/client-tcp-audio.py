# -*- coding: utf8 -*-
import cv2
import socket
import numpy as np
import threading
from queue import Queue
import time
import sys
import os
from MyVoiceRecoder import VoiceRecoder as VR
import wave
import pyaudio

HOST = 'awsbit.mynetgear.com'
PORT = 65223

## TCP 사용
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

## server ip, port
s.connect((HOST, PORT))

objVR = VR()

def gettingMsg( action ):
    while True:
        data = s.recv(1024)
        if not data: break
        else:
            data = str(data).split("b'", 1)[1].rsplit("'",1)[0]
            print(data)
            if data == 'swiping left':
                action.put( True )
            elif data == 'swiping right':
                action.put(False)

def sendVideo(VideoFrame):
    while True:
        frame = videoFrame.get()
        if videoFrame == None:
            break
        result, frame = cv2.imencode('.jpg', frame, encode_param)
        # frame을 String 형태로 변환
        data = np.array(frame)
        #stringData = data.tostring()
        stringData = data.tobytes()
	 
        #서버에 데이터 전송
	    #(str(len(stringData))).encode().ljust(16)
        s.sendall((str(len(stringData))).encode().ljust(16) + stringData)
    s.close()

def button1Function( action ):
    global objVR
    global thdRunner
    action = action.get()

    if action == False:
        print( "녹음을 시작합니다.")
        thdRunner = threading.Thread( target = objVR.run, args=() )
        print(" 음성인식을 시작합니다.")
        thdRunner.start()

        action = True
    else:
        print( "녹음중.. ")

    if action == True:
        print( "녹음 정지합니다.")
        objVR.setStopsign( True )

        while 1:
            # print( self.thdRunner.is_alive() )
            if thdRunner.is_alive() == False:
                break
        objVR.stop()
        print( "음성 인식을 종료합니다.")

        thdRunner.join()
        # print(self.thdRunner.is_alive())

        action = False

    else:
        print(" 녹음을 하지않고있습니다.")

if __name__ == '__main__':

    videoFrame = Queue()
    action = Queue()

    th1 = threading.Thread(target=gettingMsg, args=( action, ))
    th2 = threading.Thread(target=sendVideo, args=(videoFrame, ))
    th3 = threading.Thread(target=button1Function, args=(action,))

    th1.start()
    th2.start()
    th3.start()

	 
	 
    ## webcam 이미지 capture
    cam = cv2.VideoCapture(0)
	 
    ## 이미지 속성 변경 3 = width, 4 = height
    cam.set(3, 320)
    cam.set(4, 240)
	 
    ## 0~100에서 90의 이미지 품질로 설정 (default = 95)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    
    prev_time = 0
    FPS = 20
    while True:
        # 비디오의 한 프레임씩 읽는다.
        # 제대로 읽으면 ret = True, 실패면 ret = False, frame에는 읽은 프레임
        ret, frame = cam.read()
        # cv2. imencode(ext, img [, params])
        # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
        if not ret:
            break
        current_time = time.time() - prev_time
        if current_time > 1./FPS:
            prev_time = time.time()
            videoFrame.put(frame)

        key = cv2.waitKey(1) & 0xff
        if key == 'q':
            break

    th1.join()
    th2.join()
    th2.join()
    cam.release()
