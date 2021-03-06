import tensorflow as tf
import cv2
import time
import numpy as np
from PIL import ImageDraw, ImageFont, Image

##########모델 로드

labels = ['thumb_up', 'thumb_down', 'stop_sign']

model = tf.keras.models.load_model('models/hand_gestures_classification_model/saved_model')

##########모델 예측

cap = cv2.VideoCapture(0)
time.sleep(3) #warming up
if not cap.isOpened():
    exit()

images = []

while True:
    ret, image = cap.read()
    #print(type(image)) #<class 'numpy.ndarray'>
    #print(image.shape) #(720, 1280, 3)

    if not ret:
        break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    #print(rgb_image.shape) #(224, 224)
    # Our sequence length (36) of images in one video is huge and fps of camera is quite low.
    # To fill the buffer quickly, fill three images.
    images.append(rgb_image)
    images.append(rgb_image)
    images.append(rgb_image)

    if len(images) == 36:
        x_test = np.array([images])
        x_test = x_test / 255

        y_predict = model.predict(x_test)

        label = labels[y_predict[0].argmax()]
        confidence = y_predict[0][y_predict[0].argmax()]
        cv2.putText(image, text='{} {:.2f}%'.format(label, confidence * 100), org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        #pill_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #draw = ImageDraw.Draw(pill_image)
        #draw.text((30, 30), '{} {:.2f}%'.format(label, confidence * 100), font=ImageFont.truetype('malgun.ttf', 36), fill=(255, 255, 255)) #윈도우
        #draw.text((30, 30), '{} {:.2f}%'.format(label, confidence * 100), font=ImageFont.truetype('AppleGothic.ttf', 36), fill=(255, 255, 255)) #맥
        #draw.text((30, 30), '{} {:.2f}%'.format(label, confidence * 100), font=ImageFont.truetype('Ubuntu.ttf', 36), fill=(255, 255, 255)) # ubuntu
        #image = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

        images.clear()

    cv2.imshow('image', image)

    if cv2.waitKey(1) == ord('q'): #key 입력이 있을때까지 1밀리 세컨드 만큼 대기
        break

cap.release()
cv2.destroyAllWindows()
