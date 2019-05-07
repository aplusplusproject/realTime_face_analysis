
import cv2
import time
import requests
import threading
from threading import Thread, Event, ThreadError

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

cap = cv2.VideoCapture('rtsp://10.89.220.212:7754/h264_pcm.sdp')

fourcc = cap.get(cv2.CAP_PROP_FOURCC)
codec = decode_fourcc(fourcc)
print("Codec: " + codec)
while(1):
    ret, frame = cap.read()
    cv2.imshow('VIDEO', frame)
    cv2.waitKey(1)