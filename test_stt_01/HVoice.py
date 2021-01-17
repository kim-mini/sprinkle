import speech_recognition as sr


class H_VoiceRecoder:

    def __init__( self ):
        self.listMsg = []
        self.strMsg = ""
        self.wavAudio = None
        self.objRecognizer = sr.Recognizer()
        self.objMicroPhone = sr.Microphone()

        self.stop_listen = None

    def __del__( self ):
        self.stop_listen_background()

    def getMsg( self ):
        strMsg = self.strMsg
        self.strMsg = ""
        return strMsg

    def getAudio( self ):
        wavAudio = self.wavAudio
        self.wavAudio = None
        return wavAudio

    def reset( self ):
        self.strMsg = ""
        self.wavAudio = None



    def listen_background( self ):
        self.stop_listen = self.objRecognizer.listen_in_background( self.objMicroPhone, self.callback )

    def stop_listen_background( self ):
        if self.stop_listen != None:
            self.stop_listen( wait_for_stop=False )
            self.stop_listen  = None

    def callback( self, recognizer, audio ):
        try:
            strMsg = recognizer.recognize_google( audio, language='ko' )
            print( "HVoice : " + self.strMsg )
            self.strMsg = strMsg
            self.wavAudio = audio
        except Exception as e:
            print( "Exception: " + str( e ) )

