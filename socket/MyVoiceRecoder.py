import os
import sys
import time
from HVoice import H_VoiceRecoder as HV


class VoiceRecoder():

    def __init__( self ):
        self.strUsageMsg = "Usage : python MyVoiceRecoder.py [ data dir path ]"
        self.strDataDirPath = "voiceData"
        self.bStop = False

        # 저장 디렉토리가 없다면 디렉토리 생성
        if os.path.exists( self.strDataDirPath ) == False :
            os.mkdir( self.strDataDirPath )

        self.objHV = HV()


    def __exit__( self ):
        self.objHV.stop_listen_background()



    # def gethanelso( self ):
    #     result = self.bStop
    #     return result



    def run( self ):

        # 백그라운드로 마이크 입력 시작

        self.objHV.listen_background()



        while 1:

            if self.bStop == True:

                break



            strMsg = self.objHV.getMsg()

            wavAudio = self.objHV.getAudio()



            if strMsg != "":

                print( "MyVoiceRecoder : " + strMsg )

                # print( type( wavAudio ))

                # print( "hanelso: " + str(self.bStop) )

                # if self.bStop == True:

                #     break


                strTimeFormat = self.getTimeFormat()
                strFileName = self.strDataDirPath + "/" + strTimeFormat
                print( "파일저장한다.")
                self.saveDataSet( strFileName, wavAudio, strMsg )


            time.sleep( 0.1 )


    def stop( self ):
        self.objHV.__del__()
        self.objHV = HV()
        self.bStop = False


    def setStopsign( self, bStop ):
        self.bStop = bStop


    def saveWav( self, saveWavPath, wavAudio ):
        with open( saveWavPath, "wb" ) as wavTmp:
            wavTmp.write( wavAudio.get_wav_data() )

    def saveTxt( self, saveTxtPath, strMsg ):
        with open( saveTxtPath, "w" ) as txtTmp:
            txtTmp.write( strMsg )


    def saveDataSet( self, savePath, wavAudio, strMsg ):
        saveWavPath = savePath + ".wav"
        saveTxtPath = savePath + ".txt"

        self.saveWav( saveWavPath, wavAudio )
        self.saveTxt( saveTxtPath, strMsg )

    def getTimeFormat( self ):
        now = time.localtime()

        strTimeFormat = "%04d%02d%02d%02d%02d%02d"%( now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec )
        return strTimeFormat



if __name__=="__main__":
    strUsageMsg = "Usage : python MyVoiceRecoder.py [ data dir path ]"
