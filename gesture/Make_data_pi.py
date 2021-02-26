import cv2
import os
import socket
import numpy as np
import time

class Makedata():
    def __init__( self ):
        # tcp로 연결된 라즈베리파이 카메라에 연결
        self.HOST='192.168.1.4'
        self.PORT=65223
        
        # data를 저장할 파일 경로
        self.dir_path = './data/02/go_away'# stop
        if not os.path.exists( self.dir_path ):
            os.makedirs( self.dir_path )
        self.cnt = 0

    #socket에서 수신한 버퍼를 반환하는 함수
    def recvall( self, sock, count ):
        # 바이트 문자열
        buf = b''
        while count:
            newbuf = sock.recv( count )
            if not newbuf: return None
            buf += newbuf
            count -= len( newbuf )
        return buf

    def soket_Conect( self ):

        #TCP 사용
        s=socket.socket( socket.AF_INET,socket.SOCK_STREAM )
        print( 'Socket created' )

        #서버의 아이피와 포트번호 지정
        s.bind(( self.HOST,self.PORT ))
        print( 'Socket bind complete' )
        # 클라이언트의 접속을 기다린다. (클라이언트 연결을 10개까지 받는다)
        s.listen( 10 )
        print( 'Socket now listening' )

        #연결, conn에는 소켓 객체, addr은 소켓에 바인드 된 주소
        conn,addr=s.accept()
        return conn, addr

    def getData( self, cam = True ): 
        conn, addr = self.soket_Conect()
        time.sleep(3)
        fps = 0
        while True:
            length = self.recvall(conn, 16)
            stringData = self.recvall(conn, int(length))
            data = np.fromstring( stringData, dtype = 'uint8' )
            #data를 디코딩한다.
            frame = cv2.imdecode( data, cv2.IMREAD_COLOR )

            cv2.imshow( 'Video', frame )

            imageFile = os.path.join( self.dir_path, str( fps ) + '.jpg' )
            if not self.cnt%3 :
                cv2.imwrite( imageFile, frame )
                fps += 1


            if len(os.listdir( self.dir_path )) == 37:
                break

            key = cv2.waitKey( 1 ) & 0xff
            if key == 27:
                break
            self.cnt += 1


if __name__=='__main__':
    data = Makedata()
    data.getData()
    cv2.destroyAllWindows()
