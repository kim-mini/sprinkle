# https://github.com/rmorriss/lightify_scripts/blob/master/osram_control.py

#!/usr/bin/env python
from multiprocessing import Process, Queue, set_start_method
from PIL import Image
import cv2
import numpy as np
import time
import socket

import os
import errno
import torch
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
from collections import OrderedDict, deque

from model import ConvColumn
import torch.nn as nn
import json

import imutils
from imutils.video import VideoStream, FileVideoStream, WebcamVideoStream, FPS
import argparse
import pyautogui
import configparser



qsize=20
num_classes = 9
threshold = 0.7


# from train_data.classes_dict in train.py
gesture_dict = {
    'Doing other things': 0, 0: 'Doing other things',
    'No gesture': 1, 1: 'No gesture',
    'Stop Sign': 2, 2: 'Stop Sign',
    'Swiping Down': 3, 3: 'Swiping Down',
    'Swiping Left': 4, 4: 'Swiping Left',
    'Swiping Right': 5, 5: 'Swiping Right',
    'Swiping Up': 6, 6: 'Swiping Up',
    'Turning Hand Clockwise': 7, 7: 'Turning Hand Clockwise',
    'Turning Hand Counterclockwise': 8, 8: 'Turning Hand Counterclockwise'
}

# construct the argument parse and parse the arguments
str2bool = lambda x: (str(x).lower() == 'true')
parser = argparse.ArgumentParser()
# parser.add_argument('model')nppnpp
parser.add_argument("-e", "--execute", type=str2bool, default=True, help="Bool indicating whether to map output to keyboard/mouse commands")
parser.add_argument("-d", "--debug", type=str2bool, default=True, help="In debug mode, show webcam input")
parser.add_argument("-u", "--use_gpu", type=str2bool, default=True, help="Bool indicating whether to use GPU. False - CPU, True - GPU")
parser.add_argument("-g", "--gpus", default=[4], help="GPU ids to use")
# parser.add_argument("-c", "--config", default='./config.json', help="path to configuration file")
# parser.add_argument("-v", "--video", default='./gesture.mp4', help="Path to video file if using an offline file")
# parser.add_argument("-v", "--video", default='', help="Path to video file if using an offline file")
parser.add_argument("-vb", "--verbose", default=2, help="Verbosity mode. 0- Silent. 1- Print info messages. 2- Print info and debug messages")
parser.add_argument("-cp", "--checkpoint", default="./model_best.pth.tar", help="Location of model checkpoint file")
parser.add_argument("-m", "--mapping", default="./mapping.ini", help="Location of mapping file for gestures to commands")
args = parser.parse_args()

# parser.print_help()
# sys.exit(1)

print('Using %s for inference' % ('GPU' if args.use_gpu else 'CPU'))

# initialise some variables
verbose = args.verbose
device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

transform = Compose([
        ToPILImage(),
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

model = ConvColumn(num_classes)

# read in configuration file for mapping of gestures to keyboard keys
mapping = configparser.ConfigParser()
action = {}
if os.path.isfile(args.mapping):
    mapping.read(args.mapping)

    for m in mapping['MAPPING']:
        val = mapping['MAPPING'][m].split(',')
        action[m] = {'fn': val[0], 'keys': val[1:]}  # fn: hotkey, press, typewrite

else:
    # print('[ERROR] Mapping file for gestures to keyboard keys is not found at ' + args.mapping)
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.mapping)


if args.use_gpu:
    model.cuda()
    # model = torch.nn.DataParallel(model, device_ids=args.gpus).to(device)

if os.path.isfile(args.checkpoint):
    # if (verbose>0): print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    new_state_dict = OrderedDict()

    for k, v in checkpoint.items():
        if(k == 'state_dict'):
            del checkpoint['state_dict']
            for j, val in v.items():
                name = j[7:] # remove `module.`
                new_state_dict[name] = val
            checkpoint['state_dict'] = new_state_dict
            break
    # start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    if (verbose>0): print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.checkpoint, checkpoint['epoch']))
else:
    # print("[ERROR] No checkpoint found at '{}'".format(args.checkpoint))
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.checkpoint)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
if verbose>0: print("[INFO] Attemping to start video stream...")

# if (args.video == ''):
#     vs = VideoStream(0, usePiCamera=True).start()
# else:
#     vs = FileVideoStream(args.video).start()

ret = True

# socket에서 수신한 버퍼를 반환하는 함수이다
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# 이미지 출력을 한다
def image_display( VQ, conn ):
    #cv2.namedWindow ('image_display', cv2.CV_WINDOW_AUTOSIZE)
    while True:
        frame = VQ.get()              # Added
        if frame is None:  break             # Added
        cv2.imshow ('image_display', frame)  # Added
        key = cv2.waitKey(1) & 0xff  # Added
        continue
    cv2.destroyAllWindows()
    conn.close()

# 이미지를 학습해서 제스쳐를 인식한다
def image_train( outV, TQ, SQ, GQ, act):
    global ret
    while True:
        start = time.time()

        frame = outV.get()

        if frame is None:
            print('[ERROR] No video stream is available')
            break

        else:
            if ret:
                # frame = transform(frame)
                for i in range(qsize):
                    TQ.append(frame)
                if (verbose > 0): print('[INFO] Video stream started...')
                ret = False
        oframe = frame.copy()
        frame = imutils.resize(frame, height=100)
        TQ.append(frame)

        imgs = []
        for img in TQ:
            # print(img.shape)
            img = transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        data = data[None, :, :, :, :]
        target = [2]
        target = torch.tensor(target)
        data = data.to(device)

        model.eval()  # set model to eval mode
        output = model(data)

        # send to softmax layer
        output = torch.nn.functional.softmax(output, dim=1)

        k = 5
        ts, pred = output.detach().cpu().topk(k, 1, True, True)
        top5 = [gesture_dict[pred[0][i].item()] for i in range(k)]

        pi = [pred[0][i].item() for i in range(k)]
        ps = [ts[0][i].item() for i in range(k)]
        top1 = top5[0] if ps[0] > threshold else gesture_dict[0]

        hist = {}
        for i in range(num_classes):
            hist[i] = 0
        for i in range(len(pi)):
            hist[pi[i]] = ps[i]
        SQ.append(list(hist.values()))

        ave_pred = np.array(SQ).mean(axis=0)
        top1 = gesture_dict[np.argmax(ave_pred)] if max(ave_pred) > threshold else gesture_dict[0]

        if (args.debug):
            cv2.putText(oframe, top1 + ' %.2f' % ps[0], (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2,
                        lineType=cv2.LINE_AA)
            cv2.putText(oframe, top1 + ' %.2f' % ps[0], (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1,
                        lineType=cv2.LINE_AA)
            cv2.imshow('image_display', oframe)
            key = cv2.waitKey(1) & 0xff

        top1 = top1.lower()
        GQ.put(top1)
        #act.append(top1)
        act.put(top1)
        #print("act : ",act)
        end = time.time()
        # print('학습 완료시간까지 걸리는 시간 : {}'.format(end-start))



# 제스쳐 인식 되면 라즈베리로 보내준다
def sendingMsg( conn, act, GQ ):
    act_list = list()
    while True:
        basic_gesture = ['doing other things', 'no gesture']
        top1 = GQ.get()
        top1 = top1.lower()

        if top1 is None or act is None: break
        # q or esc 키를 누르면 꺼짐

        act_list.append( act.get())
       # print("act2 : ", act_list)
       # if (act[0] != act[1] and len(set(list(act)[1:])) == 1):
        if len(set(act_list)) == 2:
            if top1 in basic_gesture:
                pass
            else:
                data = top1
                data = data.encode("utf-8")
                conn.send(data)
                act_list = act_list[-1:]

        if len(act_list) == 10:
            act_list = act_list[-1:]
    conn.close()


if __name__ == '__main__':
    set_start_method('spawn', True)
    # 입력 이미지 ( image )
    VQ = Queue(maxsize=qsize)
    # 출력할 이미지
    outV = Queue(maxsize=qsize)
    # 제스쳐 ( text )
    GQ = Queue(maxsize=qsize)

    SQ = deque(maxlen=qsize)
    # 훈련 시킬 이미지 ( 사이즈 등 전처리 후 들어가게 된다 )
    TQ = deque(maxlen=qsize)
    # act = deque(['No gesture', "No gesture"], maxlen=3)
    act = Queue(maxsize=3)
    act.put('No gesture')
    act.put('No gesture')
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
    # 입력받은 이미지를 화면에 출력해주는 함수
    pV = Process(target=image_display, args=( VQ, conn, ))
    pV.start()
    # 입력받은 이미지를 훈련시키는 함수
    pTV = Process(target=image_train, args=( outV, TQ, SQ, GQ, act, ))
    pTV.start()
    # 메세지를 보내는 함수
    p1 = Process( target=sendingMsg, args=( conn, act, GQ, ))
    p1.start()

    time.sleep(2.0)
    fps = FPS().start()
    while True:
        length = recvall(conn, 16)
        stringData = recvall(conn, int(length))
        #data = np.fromstring(stringData, dtype='uint8')
        data = np.frombuffer(stringData, dtype='uint8')
        if data is None: break
        # data를 디코딩한다.
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        VQ.put(frame)
        outV.put(frame)
        time.sleep(0.010)

        key = cv2.waitKey(1) & 0xff
        if key == 'q':
            cv2.destroyAllWindows()
            break
        fps.update()

    print(1)
    VQ.put(None)
    SQ.append(None)
    GQ.put(None)
    TQ.append(None)
    outV.put(None)

    pV.join()
    pTV.join()
    p1.join()

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


