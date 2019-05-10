
import cv2

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

cap = cv2.VideoCapture('http://10.89.146.64:8080/video')
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) #set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480) #set frame height

fourcc = cap.get(cv2.CAP_PROP_FOURCC)
codec = decode_fourcc(fourcc)
print("Codec: " + codec)
c = 0
while(1):
    ret, frame = cap.read()
    cv2.imshow('im', frame)
    cap.set(cv2.CAP_PROP_POS_MSEC, 0.5 * 1000 * c)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)

    c += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import vlc
# player=vlc.MediaPlayer('rtsp://10.89.146.64:8080/h264_ulaw.sdp')
# player.play()