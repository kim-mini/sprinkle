#!/usr/bin/env python
from multiprocessing import Process, Queue
from PIL import Image
import cv2
import numpy as np
import time
import socket





qsize=20


# socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def image_display(taskqueue, conn):
    #cv2.namedWindow ('image_display', cv2.CV_WINDOW_AUTOSIZE)
    while True:
        image = taskqueue.get()              # Added
        if image is None:  break             # Added
        cv2.imshow ('image_display', image)  # Added
        key = cv2.waitKey(1) & 0xff                # Added
        continue                             # Added

    cv2.destroyAllWindows()
    conn.close()

def sendingMsg( SQ, conn ):
    while True:
        print(1)
        data = SQ.get()
        if data is None:  break
        data = data.encode( "utf-8" )
        conn.send( data )
    conn.close()

if __name__ == '__main__':

    VQ = Queue(maxsize=qsize)
    SQ = Queue(maxsize=qsize)

    HOST = '192.168.1.4'
    PORT = 65223

    # TCP 사용
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    # 서버의 아이피와 포트번호 지정
    s.bind((HOST, PORT))
    print('Socket bind complete')
    # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
    s.listen(10)
    print('Socket now listening')

    # 연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
    conn, addr = s.accept()

    pV = Process(target=image_display, args=(VQ, conn))
    pV.start()

    p1 = Process( target=sendingMsg, args=(SQ, conn))

    p1.start()
    while True:
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        #data = np.fromstring(stringData, dtype='uint8')
        data = np.frombuffer(stringData, dtype='uint8')
        # data를 디코딩한다.
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        VQ.put(image)
        time.sleep(0.010)
        continue

    VQ.put(None)
    SQ.put(None)
    pV.join()
    #p1.join()

