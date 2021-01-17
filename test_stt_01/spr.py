import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5 import uic
from test import VoiceRecoder as VR
import wave
import pyaudio
import threading

class Form(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.objVR = VR()
        self.thdRunner = threading.Thread(target=self.objVR.run)
        self.thdRunner.start()

    def initUI(self):
        x = 2000
        y = 2000


        self.web =QWebEngineView(self)
        #self.web.move(30,10)
        self.web.setGeometry(30,10,x-60,y-200)
        self.web.setUrl(QUrl('http://naver.com'))
        #self.web.setZoomFactor(1.0)

        self.sttMsg = QLabel('요청을 입력해 주세요',self)
        self.sttMsg.setAlignment(Qt.AlignCenter)
        self.sttMsg.setGeometry(30,y-150,x-60,100)
        # self.sttMsg
        self.setWindowTitle('Sprinkle Demo')
        self.setGeometry(300, 600, x, y)
        self.show()

        self.recod()


    def recod(self):

        #self.thdRunner = threading.Thread( target = self.objVR.run )
        print(" 음성인식을 시작합니다.")
        #self.thdRunner.start()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    exit(app.exec_())